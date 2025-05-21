"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
import json
import os
from typing import Optional, Dict, List
from pathlib import Path
import time
from collections import deque
import logging
from logging_config import get_logger
import copy
from encoder import CustomJSONEncoder, serialize_env_spec
from moviepy.editor import ImageSequenceClip
from umap import UMAP
import plotly.express as px

from rl_callbacks import WandbCallback, Callback
from rl_callbacks import load as callback_load
from models import select_policy_model, StochasticContinuousPolicy, StochasticDiscretePolicy, ValueModel, CriticModel, ActorModel
from schedulers import ScheduleWrapper
from adaptive_kl import AdaptiveKL
from buffer import Buffer, ReplayBuffer, PrioritizedReplayBuffer, Buffer
from normalizer import Normalizer, SharedNormalizer
from noise import Noise, NormalNoise, UniformNoise, OUNoise
import wandb
import wandb_support
from torch_utils import set_seed, get_device, move_to_device, VarianceScaling_
from env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from utils import render_video, build_env_wrapper_obj, check_for_inf_or_NaN

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta, Normal, kl_divergence
import gymnasium as gym
# import gymnasium_robotics as gym_robo
from gymnasium.envs.registration import EnvSpec
import numpy as np



from agent_utils import load_agent_from_config, get_agent_class_from_type, compute_n_step_return, compute_full_return


# Agent class
class Agent:
    """Base class for all RL agents."""

    def __init__(self, env: EnvWrapper, callbacks: Optional[list[Callback]] = None, save_dir: str = "models/", device: str | T.device = None, log_level: str = 'info'):
        try:
            self.save_dir = self._setup_save_dir(save_dir)
            self.env = env
            self.callbacks = self._initialize_callbacks(callbacks)
            self.device = get_device(device)
            self.logger = get_logger(__name__, log_level)

            # Set internal attributes
            self._distributed = False
            self._train_config = {}
            self._train_episode_config = {}
            self._train_step_config = {}
            self._test_config = {}
            self._test_episode_config = {}
            self._test_step_config = {}
            self._step = None
        except Exception as e:
            self.logger.error(f"Error in Agent init: {e}", exc_info=True)

    def _setup_save_dir(self, save_dir: str):
        """
        Setup the save directory for the agent.
        If save_dir doesn't end with the agent's name, append it.
        
        Args:
            save_dir (str): Base save directory path
        """
        agent_name = self.__class__.__name__.lower()
        if f"/{agent_name}/" not in save_dir:
            return save_dir + f"/{agent_name}/"
        else:
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

    def _distributed_learn(self, step: int = None):
        """Handle distributed learning for both on-policy and off-policy agents."""
        raise NotImplementedError("Subclasses must implement _distributed_learn.")
    
    def get_parameters(self):
        """Return a dictionary of model parameters: {model_name: params}."""
        raise NotImplementedError("Subclasses must implement get_parameters.")

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
        cloned_agent = copy.deepcopy(self)

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
                for model_key in ['actor_model', 'critic_model', 'value_model', 'policy_model']:
                    if model_key in cloned_agent._config and isinstance(cloned_agent._config[model_key], dict):
                        if 'device' in cloned_agent._config[model_key]:
                            cloned_agent._config[model_key]['device'] = device_str
            
            # Explicitly update model device attributes
            for model_name in ['actor_model', 'critic_model', 'value_model', 'policy_model']:
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
            return move_to_device(cloned_agent, target_device)
        
        return cloned_agent

    def get_action(self, state):
        """Returns an action given a state."""
        raise NotImplementedError("Subclasses must implement get_action.")

    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None
    ):
        """Trains the model for 'episodes' number of episodes."""
        raise NotImplementedError("Subclasses must implement train.")
    
    def learn(self):
        """Updates the model."""
        raise NotImplementedError("Subclasses must implement learn.")

    def test(self, num_episodes=None, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""
        raise NotImplementedError("Subclasses must implement test.")

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
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models/",
        device: str = None,
        log_level: str = 'info'
    ):
        super().__init__(env, callbacks, save_dir, device, log_level)
        self.policy_model = policy_model
        self.value_model = value_model
        self.discount = discount
        self.policy_trace_decay = policy_trace_decay
        self.value_trace_decay = value_trace_decay
        # self.callbacks = callbacks
        # self.save_dir = save_dir
        if save_dir is not None and "/actor_critic/" not in save_dir:
            self.save_dir = save_dir + "/actor_critic/"
        elif save_dir is not None:
            self.save_dir = save_dir

        # Initialize callback configurations
        self._initialize_callbacks(callbacks)
        self._train_config = {}
        self._train_episode_config = {}
        self._train_step_config = {}
        self._test_config = {}
        self._test_step_config = {}
        self._test_episode_config = {}

        self._step = None
        # set self.action to None
        self.action = None
        # instantiate and set policy and value traces
        self.policy_trace = []
        self.value_trace = []
        self._set_traces()

    # def _initialize_callbacks(self, callbacks):
    #     """
    #     Initialize and configure callbacks for logging and monitoring.

    #     Args:
    #         callbacks (list): List of callback objects.
    #     """
    #     try:
    #         self.callbacks = callbacks
    #         if callbacks:
    #             for callback in self.callbacks:
    #                 self._config = callback._config(self)
    #                 if isinstance(callback, WandbCallback):
    #                     self._wandb = True
    #         else:
    #             self.callback_list = None
    #             self._wandb = False
    #     except Exception as e:
    #         logger.error(f"Error initializing callbacks: {e}", exc_info=True)

#     @classmethod
#     def build(
#         cls,
#         env,
#         policy_layers,
#         value_layers,
#         callbacks,
#         config,#: wandb.config,
#         save_dir: str = "models/",
#     ):
#         """Builds the agent."""
#         policy_optimizer = wandb.config.policy_optimizer
#         value_optimizer = wandb.config.value_optimizer
#         policy_learning_rate = wandb.config.learning_rate
#         value_learning_rate = wandb.config.learning_rate
#         policy_model = models.StochasticDiscretePolicy(
#             env, dense_layers=policy_layers, optimizer=policy_optimizer, learning_rate=policy_learning_rate
#         )
#         value_model = models.ValueModel(
#             env, dense_layers=value_layers, optimizer=value_optimizer, learning_rate=value_learning_rate
#         )

#         agent = cls(
#             env,
#             policy_model,
#             value_model,
#             config.discount,
#             config.policy_trace_decay,
#             config.value_trace_decay,
#             callbacks,
#             save_dir=save_dir,
#         )

#         agent.save()

#         return agent

    def _set_traces(self):
        for weights in self.policy_model.parameters():
            self.policy_trace.append(T.zeros_like(weights, device=self.device))
            #DEBUG
            # print(f'policy trace shape: {weights.size()}')

        for weights in self.value_model.parameters():
            self.value_trace.append(T.zeros_like(weights, device=self.device))
            #DEBUG
            # print(f'value trace shape: {weights.size()}')


    def _update_traces(self):
        with T.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                self.policy_trace[i] = (self.discount * self.policy_trace_decay * self.policy_trace[i]) + weights.grad

            for i, weights in enumerate(self.value_model.parameters()):
                self.value_trace[i] = (self.discount * self.value_trace_decay * self.value_trace[i]) + weights.grad

        # log to train step
        # for i, (v_trace, p_trace) in enumerate(zip(self.value_trace, self.policy_trace)):
            # self._train_step_config[f"value trace {i}"] = T.histc(v_trace, bins=20)
            # self._train_step_config[f"policy trace {i}"] = T.histc(p_trace, bins=20)

    def get_action(self, states):
        # state =  T.from_numpy(state).to(self.device)
        states = T.tensor(states, dtype=T.float32, device=self.policy_model.device)
        dist, logits = self.policy_model(states)
        actions = dist.sample()
        # log_probs = dist.log_prob(actions)
        
        actions = actions.detach().cpu().numpy() # Detach from graph, move to CPU and convert to numpy for Gym
        return actions, dist, logits

    def train(self, num_episodes, num_envs: int, seed: int | None = None, render_freq: int = 0):
        """Trains the model for 'episodes' number of episodes."""
        # set models to train mode
        self.policy_model.train()
        self.value_model.train()

        self.num_envs = num_envs

        if seed is None:
            seed = np.random.randint(100)

        # Set seeds
        set_seed(seed)

        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, WandbCallback):
                    self._config['num_episodes'] = num_episodes
                    self._config['seed'] = seed
                    self._config['num_envs'] = self.num_envs
                    callback.on_train_begin((self.policy_model, self.value_model,), logs=self._config)
                else:
                    callback.on_train_begin(logs=self._config)
        
        try:
            # instantiate new vec environment
            self.env.env = self.env._initialize_env(0, self.num_envs, seed)
        except Exception as e:
            self.logger.error(f"Error in ActorCritic.train self.env._initialize_env process: {e}", exc_info=True)

        # set step counter
        self._step = 0
        # Instantiate counter to keep track of number of episodes completed
        self.completed_episodes = 0
        # set best reward
        best_reward = -np.inf
        # Instantiate a deque to track last 10 scores for computing avg
        completed_scores = deque(maxlen=10)
        episode_scores = np.zeros(self.num_envs)
        states, _ = self.env.reset()
        
        while self.completed_episodes < num_episodes:
            # Increment step counter
            self._step += 1
            # if self.callbacks:
            #     for callback in self.callbacks:
            #         callback.on_train_epoch_begin(epoch=self._step, logs=None)
            
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_begin(step=self._step, logs=None)
            
            actions, dist, logits = self.get_action(states)
            # actions, log_probs = self.get_action(states)
            actions = self.env.format_actions(actions)
            next_states, rewards, terms, truncs, _ = self.env.step(actions)
            self._train_step_config["step_reward"] = rewards.mean()
            # self._train_step_config["logits"] = wandb.Histogram(logits.detach().cpu().numpy())
            episode_scores += rewards
            dones = np.logical_or(terms, truncs)
            self.learn(states, rewards, next_states, actions, dist, dones)
            # self.learn(states, rewards, next_states, actions, log_probs, dones)

            for i in range(self.num_envs):
                if dones[i]:
                    # Increment completed episodes
                    self.completed_episodes += 1
                    # Append episode score to completed scores deque
                    completed_scores.append(episode_scores[i])
                    self._train_episode_config["episode_reward"] = episode_scores[i]
                    self._train_episode_config["episode"] = self.completed_episodes
                    # Reset episode score
                    episode_scores[i] = 0
                    avg_reward = sum(completed_scores) / len(completed_scores)

                    # check if best reward
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        self._train_episode_config["best"] = 1
                        # save model
                        self.save()
                    else:
                        self._train_episode_config["best"] = 0

                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

                    # Check if number of completed episodes should trigger render
                    if render_freq > 0:
                        if self.completed_episodes % render_freq == 0 and not rendered:
                            print(f"Rendering episode {self.completed_episodes} during training...")
                            # Call the test function to render an episode
                            self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes}.mp4")
                            # Log the video to wandb
                            if self.callbacks:
                                for callback in self.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")}, step=self._step)
                            rendered = True
                            # Switch models back to train mode after rendering
                            self.policy_model.train()
                            self.value_model.train()
                        else:
                            rendered = False

                    print(f"episode {self.completed_episodes}, score {completed_scores[-1]}, avg_score {avg_reward}")

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=self._step, logs=self._train_step_config)
            
            states = next_states

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)

    def learn(self, states, rewards, next_states, actions, dist, dones):
        self.policy_model.optimizer.zero_grad()
        self.value_model.optimizer.zero_grad()

        states = T.tensor(states, dtype=T.float32, device=self.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.device)
        next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
        actions = T.tensor(actions, dtype=T.long, device=self.device)
        dones = T.tensor(dones, dtype=T.int, device=self.device)
        # Run states through policy model to get distribution and log probs of actions
        # dist, logits = self.policy_model(states)
        log_probs = dist.log_prob(actions)

        state_values = self.value_model(states)
        next_state_values = self.value_model(next_states).detach()
        temporal_difference = (
            rewards + self.discount * next_state_values.squeeze() * (1 - dones) - state_values.squeeze())
        value_loss = temporal_difference.square().mean()

        value_loss.backward()

        policy_loss = -(log_probs * temporal_difference.detach()).mean()
        policy_loss.backward()

        # total_loss = value_loss + policy_loss
        # total_loss.backward()

        self._update_traces()

        #DEBUG
        # print(f'dones:{dones}')
        # print(f'states size: {states.size()}')
        # print(f'next states size: {next_states.size()}')
        # print(f'rewards size: {rewards.size()}')
        # print(f'actions size: {actions.size()}')
        # print(f'dones size: {dones.size()}')
        # print(f'log_probs size: {log_probs.size()}')
        # print(f'state values size: {state_values.size()}')
        # print(f'squeezed state values:{state_values.squeeze().size()}')
        # print(f'squeezed next state values:{next_state_values.squeeze().size()}')
        # print(f'next state values size: {next_state_values.size()}')
        # print(f'temporal_difference size: {temporal_difference.size()}')
        # print(f'value loss size: {value_loss.size()}')
        # print(f'policy loss size: {policy_loss.size()}')

        #copy traces to weight gradients
        with T.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                weights.grad = self.policy_trace[i]

            for i, weights in enumerate(self.value_model.parameters()):
                weights.grad = self.value_trace[i]

        self.value_model.optimizer.step()
        self.policy_model.optimizer.step()

        
        self._train_step_config["policy_loss"] = policy_loss.item()
        self._train_step_config["value_loss"] = value_loss.item()
        self._train_step_config["temporal_difference"] = temporal_difference.mean()
        # self._train_step_config[f"logits"] = wandb.Histogram(logits.detach().cpu().numpy())
        self._train_step_config[f"actions"] = wandb.Histogram(actions.detach().cpu().numpy())
        self._train_step_config[f"log_probabilities"] = wandb.Histogram(log_probs.detach().cpu().numpy())
        self._train_step_config[f"action_probabilities"] = wandb.Histogram(dist.probs.detach().cpu().numpy())
        self._train_step_config["entropy"] = dist.entropy().mean().item()

    def test(self, num_episodes:int, num_envs: int=1, seed: int=None, render_freq: int=0, training: bool=False):
        """Runs a test over 'num_episodes'."""
        # Set models to eval mode
        self.policy_model.eval()
        self.value_model.eval()

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        if render_freq == None:
            render_freq = 0

        # Set seeds
        set_seed(seed)

        try:
            # instantiate new vec environment
            env = self.env._initialize_env(render_freq, num_envs, seed)
        except Exception as e:
            self.logger.error(f"Error in ActorCritic.test agent._initialize_env process: {e}", exc_info=True)

        if self.callbacks and not training:
            print('test begin callback if statement fired')
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    # Add to config to send to wandb for logging
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)
        completed_episodes = 0
        # Instantiate array to keep track of current episode scores
        episode_scores = np.zeros(num_envs)
        # Instantiate a deque to track last 'episodes_per_update' scores for computing avg
        completed_scores = deque(maxlen=num_episodes)
        # Instantiate list to keep track of frames for rendering
        frames = []
        states, _ = env.reset()
        _step = 0
            
        while completed_episodes < num_episodes:
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_begin(epoch=_step, logs=None)
            actions, _, _ = self.get_action(states)
            actions = self.env.format_actions(actions)
            next_states, rewards, terms, truncs, _ = env.step(actions)
            episode_scores += rewards
            dones = np.logical_or(terms, truncs)

            for i in range(self.num_envs):
                if dones[i]:
                    completed_scores.append(episode_scores[i])
                    # Add the episode reward to the episode log for callbacks
                    self._test_episode_config["episode_reward"] = episode_scores[i]
                    # Reset the episode score of the env back to 0
                    episode_scores[i] = 0
                    # check if best reward
                    avg_reward = sum(completed_scores) / len(completed_scores)
                    # Increment completed_episodes counter
                    completed_episodes += 1
                    # Log completed episodes to callback episode config
                    self._test_episode_config["episode"] = completed_episodes
                    # Save the video if the episode number is divisible by render_freq
                    if (render_freq > 0) and ((completed_episodes) % render_freq == 0):
                        if training:
                            render_video(frames, self.completed_episodes, self.save_dir, 'train')
                        else:
                            render_video(frames, completed_episodes, self.save_dir, 'test')
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/test/episode_{completed_episodes}.mp4")
                            # Log the video to wandb
                            if self.callbacks:
                                for callback in self.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        wandb.log({"training_video": wandb.Video(video_path, caption="Testing process", format="mp4")})
                        # Empty frames array
                        frames = []
                    # Signal to all callbacks that an episode (epoch) has completed and to log data
                    if self.callbacks and not training:
                        for callback in self.callbacks:
                            callback.on_test_epoch_end(
                            epoch=_step, logs=self._test_episode_config
                        )
                    if not training:
                        print(f"episode {completed_episodes}, score {completed_scores[-1]}, avg_score {avg_reward}")

            if render_freq > 0:
                # Capture the frame
                frame = env.render()[0]
                # print(f'frame:{frame}')
                frames.append(frame)

            states = next_states

            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=_step, logs=self._test_step_config)

        if self.callbacks and not training:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "discount": self.discount,
            "policy_trace_decay": self.policy_trace_decay,
            "value_trace_decay": self.value_trace_decay,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir
        }


    def save(self):
        """Saves the model."""
        config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of actor critic agent config
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        # saves policy and value model
        self.policy_model.save(self.save_dir)
        self.value_model.save(self.save_dir)

        # # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")

    @classmethod
    def load(cls, config, load_weights=True):
        """Loads the model."""
        # Load EnvWrapper
        env_wrapper = EnvWrapper.from_json(config["env"])
        # load policy model
        policy_model = StochasticDiscretePolicy.load(config['save_dir'], load_weights)
        # load value model
        value_model = ValueModel.load(config['save_dir'], load_weights)
        # load callbacks
        callbacks = [callback_load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]\
                    if config['callbacks'] else None

        # return Actor-Critic agent
        agent = cls(
            env=env_wrapper,
            policy_model=policy_model,
            value_model=value_model,
            discount=config["discount"],
            policy_trace_decay=config["policy_trace_decay"],
            value_trace_decay=config["value_trace_decay"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
        )

        return agent


class Reinforce(Agent):
    def __init__(
        self,
        env: EnvWrapper,
        policy_model: StochasticDiscretePolicy,
        value_model: Optional[ValueModel] = None,
        discount: float = 0.99,
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: str = None,
    ):
        # Set the device
        self.device = get_device(device)

        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.discount = discount
        # self.callbacks = callbacks
        # self.save_dir = save_dir
        if save_dir is not None and "/reinforce/" not in save_dir:
            self.save_dir = save_dir + "/reinforce/"
        elif save_dir is not None:
            self.save_dir = save_dir

        self._step = None
        self._cur_learning_steps = None
            
        # Initialize callback configurations
        self._initialize_callbacks(callbacks)
        self._train_config = {}
        self._train_episode_config = {}
        self._train_step_config = {}
        self._test_config = {}
        self._test_step_config = {}
        self._test_episode_config = {}

    def _initialize_callbacks(self, callbacks):
        """
        Initialize and configure callbacks for logging and monitoring.

        Args:
            callbacks (list): List of callback objects.
        """
        try:
            self.callbacks = callbacks
            if callbacks:
                for callback in self.callbacks:
                    self._config = callback._config(self)
                    if isinstance(callback, WandbCallback):
                        self._wandb = True
            else:
                self.callback_list = None
                self._wandb = False
        except Exception as e:
            logger.error(f"Error initializing callbacks: {e}", exc_info=True)

    # @classmethod
    # def build(
    #     cls,
    #     env,
    #     policy_layers,
    #     value_layers,
    #     callbacks,
    #     config,#: wandb.config,
    #     save_dir: str = "models/",
    # ):
    #     """Builds the agent."""
    #     policy_optimizer = config.policy_optimizer
    #     value_optimizer = config.value_optimizer
    #     policy_model = StochasticDiscretePolicy(
    #         env, dense_layers=policy_layers, optimizer=policy_optimizer, learning_rate=config.learning_rate
    #     )
    #     value_model = ValueModel(
    #         env, dense_layers=value_layers, optimizer=value_optimizer, learning_rate=config.learning_rate
    #     )

    #     agent = cls(
    #         env,
    #         policy_model,
    #         value_model,
    #         config.discount,
    #         callbacks,
    #         save_dir=save_dir,
    #     )

    #     agent.save()

    #     return agent

    def get_return(self, trajectories):
        """Compute expected returns per timestep for each trajectory."""

        for trajectory in trajectories:
            _return = 0.0
            for step in reversed(trajectory):
                _return = step["reward"] + self.discount * _return
                step["return"] = _return

        return trajectories
    
    def get_action(self, state):
        # with T.no_grad():
        state = T.tensor(state, dtype=T.float32, device=self.policy_model.device)
        dist, logits = self.policy_model(state)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Reshape if flat
        # if actions.dim() == 1:  # Adjust for scalar action
        #     actions = actions.unsqueeze(0)

        actions = actions.detach().cpu().numpy()
        # log_probs = log_probs.detach().cpu().numpy() # Keep as tensor for gradient
        
        
        
        return actions, log_probs

    def learn(self, trajectories):
        # Clear gradients
        self.policy_model.optimizer.zero_grad()
        self.value_model.optimizer.zero_grad()

        # Get returns for each step in all trajectories
        trajectories = self.get_return(trajectories)
            
        # Flatten trajectories across all envs
        all_steps = [step for trajectory in trajectories for step in trajectory]
        # DEBUG
        # print(f'all_steps:{all_steps}')
        # print(f'length of all_steps:{len(all_steps)}')
        # Extract states, returns, and log probs from trajectories
        all_states = [step["state"] for step in all_steps]
        all_returns = [step["return"] for step in all_steps]
        all_actions = [step["action"] for step in all_steps]
        # all_log_probs = [step["log_prob"] for step in all_steps]
        # Convert to tensors and store on correct device
        states = T.tensor(all_states, dtype=T.float32, device=self.device)
        returns = T.tensor(all_returns, dtype=T.float32, device=self.device)
        actions = T.tensor(all_actions, dtype=T.long, device=self.device)
        # log_probs = torch.stack(all_log_probs, dim=0)
        # Compute log probs by passing states through policy model, re-computing the
        # computational graph only using the log_probs for the batch update (detaches log_probs from
        # all envs in the env vector)
        dist, logits = self.policy_model(states)
        log_probs = dist.log_prob(actions)
        #DEBUG
        print(f'states shape:{states.size()}')
        # print(f'returns shape:{returns.size()}')
        # print(f'log probs shape:{log_probs.size()}')
        # print(f'log probs mean:{log_probs.mean()}')
        

        # Insert dimension at end of states if flat
        if states.dim() == 1:
            states = states.unsqueeze(-1)
            #DEBUG
            print(f'new state shape:{states.size()}')

        # Get state values and calculate value loss
        if self.value_model:
            state_values = self.value_model(states)
            advantages = returns - state_values.squeeze(-1)
            value_loss = (advantages ** 2).mean()
        else:
            advantages = returns
            value_loss = 0
        #DEBUG
        # print(f'advantages shape:{advantages.size()}')
        
        # Get policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Calculate gradients
        total_loss = policy_loss + value_loss
        total_loss.backward()
        # if self.value_model:
        #     value_loss.backward()

        # Update weights
        self.policy_model.optimizer.step()
        if self.value_model:
            self.value_model.optimizer.step()

        # log the metrics for callbacks
        # if self._wandb:
        self._train_step_config["advantages"] = advantages.mean()
        self._train_step_config["policy_loss"] = policy_loss.item()
        self._train_step_config["value_loss"] = value_loss.item()
        self._train_step_config[f"logits"] = wandb.Histogram(logits.detach().cpu().numpy())
        self._train_step_config[f"actions"] = wandb.Histogram(actions.detach().cpu().numpy())
        self._train_step_config[f"log_probabilities"] = wandb.Histogram(log_probs.detach().cpu().numpy())
        self._train_step_config[f"action_probabilities"] = wandb.Histogram(dist.probs.detach().cpu().numpy())
        self._train_step_config["entropy"] = dist.entropy().mean().item()

    def train(self, num_episodes: int, num_envs: int, trajectories_per_update: int=10, seed: int | None = None, render_freq: int = 0):
        """Trains the model for 'episodes' number of episodes."""

        # set models to train mode
        self.policy_model.train()
        self.value_model.train()

        # set num_envs as attribute
        self.num_envs = num_envs

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        # if render_freq == None:
        #     render_freq = 0

        # Set seeds
        set_seed(seed)

        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, WandbCallback):
                    self._config['num_episodes'] = num_episodes
                    self._config['seed'] = seed
                    self._config['num_envs'] = self.num_envs
                    callback.on_train_begin((self.policy_model, self.value_model,), logs=self._config)
        
        try:
            # instantiate new vec environment
            self.env.env = self.env._initialize_env(0, self.num_envs, seed)
        except Exception as e:
            logger.error(f"Error in Reinforce.train self.env._initialize_env process: {e}", exc_info=True)

        # set step counter
        self._step = 0
        best_reward = -np.inf
        self.completed_episodes = 0
        episode_scores = np.zeros(self.num_envs)

        # Instantiate a deque to track last 10 scores for computing avg
        completed_scores = deque(maxlen=10)
        # Instantiate a list of num_envs lists to store trajectories
        episode_trajectories = [[] for _ in range(self.num_envs)]
        # Instantiate a list to store completed trajectories
        completed_trajectories = []
        # Reset all envs to get initial states
        states, _ = self.env.reset()
        # Set rendered flag to trigger episode rendering during training
        rendered = False
        while self.completed_episodes < num_episodes:
            # Increment step counter
            self._step += 1
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            
            # if self.callbacks:
            #     for callback in self.callbacks:
            #         callback.on_train_step_begin(step=self._step, logs=None)
            
            # self._cur_learning_steps.append(self._step)
            actions, _ = self.get_action(states)
            actions = self.env.format_actions(actions)
            #DEBUG
            # print(f'actions pre convert:{actions}')
            # Convert actions to list of ints
            # actions = [int(a) for a in actions]
            #DEBUG
            # print(f'actions post convert:{actions}')
            next_states, rewards, terms, truncs, _ = self.env.step(actions)
            # Add data to train step log
            # self._train_step_config["actions"] = actions
            self._train_step_config["step_rewards"] = rewards.mean()
            # self._train_step_config["log_probabilities"] = log_probs
            # Add rewards to episode scores
            episode_scores += rewards
            # store trajectories
            for i in range(self.num_envs):
                episode_trajectories[i].append(
                    {
                        "state": states[i],
                        "action": actions[i],
                        # "log_prob": log_probs[i],
                        "reward": rewards[i],
                        "done": terms[i] or truncs[i]
                    }
                )
                # Perform updates if term or trunc
                if terms[i] or truncs[i]:
                    # Append episode trajectory to completed trajectories
                    completed_trajectories.append(episode_trajectories[i])
                    # Append environment score to completed scores
                    completed_scores.append(episode_scores[i])
                    # Add the episode reward to the episode log for callbacks
                    self._train_episode_config["episode_reward"] = episode_scores[i]
                    # Reset the episode score of the env back to 0
                    episode_scores[i] = 0
                    # check if best reward
                    avg_reward = sum(completed_scores) / len(completed_scores)
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        self._train_episode_config["best"] = 1
                        # save model
                        self.save()
                    else:
                        self._train_episode_config["best"] = 0
                    
                    # Reset env trajectory
                    episode_trajectories[i] = []
                    # Increment completed_episodes counter
                    self.completed_episodes += 1
                    # Log completed episodes to callback episode config
                    self._train_episode_config["episode"] = self.completed_episodes
                    # Signal to all callbacks that an episode (epoch) has completed and to log data
                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_train_epoch_end(
                            epoch=self._step, logs=self._train_episode_config
                        )
                    # Check if number of completed episodes should trigger render
                    if self.completed_episodes % render_freq == 0 and not rendered:
                        print(f"Rendering episode {self.completed_episodes} during training...")
                        # Call the test function to render an episode
                        self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
                        # Add render to wandb log
                        video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes}.mp4")
                        # Log the video to wandb
                        if self.callbacks:
                            for callback in self.callbacks:
                                if isinstance(callback, WandbCallback):
                                    wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")})
                        rendered = True
                        # Switch models back to train mode after rendering
                        self.policy_model.train()
                        self.value_model.train()
                    else:
                        rendered = False
                    # Print episode update to console
                    print(f"episode {self.completed_episodes}/{num_episodes} score: {completed_scores[-1]} avg score: {avg_reward}")

            states = next_states

            # Perform an update if the number of completed trajectories is greater than or
            # equal to the number of trajectories per update
            if len(completed_trajectories) >= trajectories_per_update:
                self.learn(completed_trajectories)
                # Clear completed_trajectories
                completed_trajectories = []

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=self._step, logs=self._train_step_config)

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)
        # close the environment
        # self.env.close()

    def test(self, num_episodes: int, num_envs: int=1, seed: int=None, render_freq: int=0, training: bool=False):
        """Runs a test over 'num_episodes'."""

        # Set models to eval mode
        self.policy_model.eval()
        self.value_model.eval()

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        if render_freq == None:
            render_freq = 0

        # Set seeds
        set_seed(seed)

        try:
            # instantiate new vec environment
            env = self.env._initialize_env(render_freq, num_envs, seed)
        except Exception as e:
            logger.error(f"Error in Reinforce.test agent._initialize_env process: {e}", exc_info=True)

        if self.callbacks and not training:
            print('test begin callback if statement fired')
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    # Add to config to send to wandb for logging
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)

        _step = 0
        completed_episodes = 0
        # Instantiate array to keep track of current episode scores
        episode_scores = np.zeros(num_envs)
        # Instantiate a deque to track last 'episodes_per_update' scores for computing avg
        completed_scores = deque(maxlen=num_episodes)
        # Instantiate list to keep track of frames for rendering
        frames = []
        # Reset environment to get starting state
        states, _ = env.reset()
        while completed_episodes < num_episodes:
            # Increment step counter
            _step += 1
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_begin(epoch=_step, logs=None)
            
            # if self.callbacks:
            #     for callback in self.callbacks:
            #         callback.on_train_step_begin(step=self._step, logs=None)
            
            # self._cur_learning_steps.append(self._step)
            actions, _ = self.get_action(states)
            actions = self.env.format_actions(actions)
            #DEBUG
            # print(f'actions pre convert:{actions}')
            # Convert actions to list of ints
            # actions = [int(a) for a in actions]
            #DEBUG
            # print(f'actions post convert:{actions}')
            next_states, rewards, terms, truncs, _ = env.step(actions)
            # Add data to train step log
            # self._train_step_config["actions"] = actions
            self._train_step_config["step_rewards"] = rewards
            # self._train_step_config["log_probabilities"] = log_probs
            # Add rewards to episode scores
            episode_scores += rewards
            # store trajectories
            for i in range(num_envs):
                # episode_trajectories[i].append(
                #     {
                #         "state": states[i],
                #         "action": actions[i],
                #         # "log_prob": log_probs[i],
                #         "reward": rewards[i],
                #         "done": terms[i] or truncs[i]
                #     }
                # )
                # Perform updates if term or trunc
                if terms[i] or truncs[i]:
                    # Append episode trajectory to completed trajectories
                    # completed_trajectories.append(episode_trajectories[i])
                    # Append environment score to completed scores
                    completed_scores.append(episode_scores[i])
                    # Add the episode reward to the episode log for callbacks
                    self._test_episode_config["episode_reward"] = episode_scores[i]
                    # Reset the episode score of the env back to 0
                    episode_scores[i] = 0
                    # check if best reward
                    avg_reward = sum(completed_scores) / len(completed_scores)
                    # if avg_reward > best_reward:
                    #     best_reward = avg_reward
                    #     self._test_episode_config["best"] = 1
                    #     # save model
                    #     self.save()
                    # else:
                    #     self._test_episode_config["best"] = 0
                    
                    # Reset env trajectory
                    # episode_trajectories[i] = []
                    # Increment completed_episodes counter
                    completed_episodes += 1
                    # Log completed episodes to callback episode config
                    self._test_episode_config["episode"] = completed_episodes
                    # Save the video if the episode number is divisible by render_freq
                    if (render_freq > 0) and ((completed_episodes) % render_freq == 0):
                        if training:
                            render_video(frames, self.completed_episodes, self.save_dir, 'train')
                        else:
                            render_video(frames, completed_episodes, self.save_dir, 'test')
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/test/episode_{completed_episodes}.mp4")
                            # Log the video to wandb
                            if self.callbacks:
                                for callback in self.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        wandb.log({"training_video": wandb.Video(video_path, caption="Testing process", format="mp4")})
                        # Empty frames array
                        frames = []
                    # Signal to all callbacks that an episode (epoch) has completed and to log data
                    if self.callbacks and not training:
                        for callback in self.callbacks:
                            callback.on_test_epoch_end(
                            epoch=_step, logs=self._test_episode_config
                        )
                    if not training:
                        # Print episode update to console
                        print(f"episode {completed_episodes}/{num_episodes} score: {completed_scores[-1]} avg score: {avg_reward}")
                
            if render_freq > 0:
                # Capture the frame
                frame = env.render()[0]
                # print(f'frame:{frame}')
                frames.append(frame)

            states = next_states

            # Perform an update if the number of completed trajectories is greater than or
            # equal to the number of trajectories per update
            # if len(completed_trajectories) >= trajectories_per_update:
            #     self.learn(completed_trajectories)
            #     # Clear completed_trajectories
            #     completed_trajectories = []

            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=_step, logs=self._test_step_config)

        if self.callbacks and not training:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)
        # close the environment
        # self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "discount": self.discount,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir
        }

    def save(self):
        """Saves the model."""
        config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of reinforce agent config
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        # saves policy and value model
        self.policy_model.save(self.save_dir)
        if self.value_model:
            self.value_model.save(self.save_dir)

        # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")

    @classmethod
    def load(cls, config, load_weights):
        """Loads the model."""
        # # load reinforce agent config
        # with open(
        #     Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
        # ) as f:
        #     obj_config = json.load(f)

        env_wrapper = EnvWrapper.from_json(config["env"])

        # load policy model
        policy_model = StochasticDiscretePolicy.load(config['save_dir'], load_weights)
        if config["value_model"]:
            # load value model
            value_model = ValueModel.load(config['save_dir'], load_weights)
        # load callbacks
        callbacks = [callback_load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]\
                    if config['callbacks'] else None

        # return reinforce agent
        agent = cls(
            env=env_wrapper,
            policy_model=policy_model,
            value_model=value_model,
            discount=config["discount"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
        )

        return agent
    

class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        env: EnvWrapper,
        actor_model: ActorModel,
        critic_model: CriticModel,
        replay_buffer: Buffer = None,
        discount: float=0.99,
        tau: float=0.001,
        action_epsilon: float = 0.0,
        batch_size: int = 64,
        noise: Noise=None,
        noise_schedule: ScheduleWrapper=None,
        grad_clip: float=None,
        warmup: int=1000,
        N: int=1, # N-steps
        callbacks: Optional[list[Callback]] = None,
        save_dir: str = "models",
        device: str = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, save_dir, device, log_level)
            self.actor_model = actor_model
            self.critic_model = critic_model
            # set target actor and critic models
            self.target_actor_model = self.clone_model(actor_model)
            self.target_critic_model = self.clone_model(critic_model)
            self.discount = discount
            self.tau = tau
            self.action_epsilon = action_epsilon
            self.replay_buffer = replay_buffer
            self.batch_size = batch_size
            self.noise = noise
            self.noise_schedule = noise_schedule
            self.grad_clip = grad_clip
            self.warmup = warmup
            self.N = N
            # logger.debug(f"rank {self.rank} DDPG init attributes set")
        except Exception as e:
            self.logger.error(f"Error in DDPG init: {e}", exc_info=True)
        
        # set internal attributes
        try:
            obs_space = (self.env.single_observation_space if hasattr(self.env, "single_observation_space") 
                        else self.env.observation_space)
            # Check if the observation space is a dictionary for goal-aware environments
            if isinstance(obs_space, gym.spaces.Dict):
                shape = obs_space['observation'].shape
                # goal_shape = obs_space['desired_goal'].shape
                # shape = (observation_shape[0] + goal_shape[0],)
            else:
                shape = obs_space.shape

            # if self.normalize_inputs:
            #     # self.state_normalizer = Normalizer(shape, self.normalizer_eps, self.normalizer_clip, self.device)
            #     self.state_normalizer = nn.BatchNorm1d(num_features=shape[-1], device=self.device)

            # Instantiate internal attribute use_her to be switched by HER class if using DDPG
            self._use_her = False

            # Set learn_iter and sync_iter to 0. For distributed training
            self._learn_iter = 0
            self._sync_iter = 0

        except Exception as e:
            self.logger.error(f"Error in DDPG init internal attributes: {e}", exc_info=True)


    # def clone(self, device: Optional[str | T.device] = None):
    #     """Clone the DDPG agent."""
    #     if device:
    #         device = get_device(device)
    #     else:
    #         device = self.device

    #     env = GymnasiumWrapper(self.env.env_spec)
    #     actor = self.clone_model(self.actor_model, device)
    #     critic = self.clone_model(self.critic_model, device)
    #     replay_buffer = self.replay_buffer.clone(device)
    #     noise = self.noise.clone()
    #     noise_schedule = ScheduleWrapper(self.noise_schedule.get_config()) if self.noise_schedule else None

    #     return DDPG(
    #         env,
    #         actor,
    #         critic,
    #         replay_buffer,
    #         self.discount,
    #         self.tau,
    #         self.action_epsilon,
    #         self.batch_size,
    #         noise,
    #         noise_schedule,
    #         self.normalize_inputs,
    #         self.normalizer_clip,
    #         self.normalizer_eps,
    #         self.warmup,
    #         self.callbacks,
    #         self.save_dir,
    #         device,
    #         logging.getLevelName(self.logger.getEffectiveLevel()).lower()
    #     )
        

    def clone_model(self, model, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Clones a model."""
        if device:
            device = get_device(device)
        else:
            device = self.device

        return model.clone(copy_weights, device)
    
    def _initialize_wandb(self, run_number:str=None, run_name_prefix:str=None, learn_iter:int=None):
        """Initialize WandbCallback if using WandbCallback"""
        try:
            if self._wandb:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if not callback.initialized:
                            models = (self.actor_model, self.critic_model)
                            config = self.get_config()
                            if learn_iter:
                                self._learn_iter = learn_iter
                                config['learn_interval'] = learn_iter
                            callback.initialize_run(models, config, run_number=run_number, run_name_prefix=run_name_prefix)
        except Exception as e:
            self.logger.error(f"Error in _initialize_wandb: {e}", exc_info=True)

    def _init_her(self):
            self._use_her = True

    def _distributed_learn(self, step: int, run_number:str=None, learn_iter:int=None):
        """Used in distributed training to update the shared models.
        This function is overridden by the Worker class to point to the Learner class.
        """
        previous_step = self._step
        # Set current step to step if greater than current step
        if step > previous_step:
            self._step = step
            # Initialize wandb check
            self._initialize_wandb(run_number=run_number, run_name_prefix="train", learn_iter=learn_iter)
            actor_loss, critic_loss = self.learn()
            # Only store log if current step greater than previous and self._wandb
            if self._wandb:
                self._train_step_config["actor_loss"] = actor_loss
                self._train_step_config["critic_loss"] = critic_loss
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        callback.on_train_step_end(step, self._train_step_config)
        else:
            actor_loss, critic_loss = self.learn()

    # def get_parameters(self):
    #     """Get the parameters of all models."""
    #     return {
    #         'actor_model': self.actor_model.state_dict(),
    #         'critic_model': self.critic_model.state_dict(),
    #         'target_actor_model': self.target_actor_model.state_dict(),
    #         'target_critic_model': self.target_critic_model.state_dict(),
    #     }

    def get_parameters(self):
        """Get the parameters of all models, ensuring they are on CPU for Ray serialization."""
        return {
            'actor_model': {k: v.cpu() for k, v in self.actor_model.state_dict().items()},
            'critic_model': {k: v.cpu() for k, v in self.critic_model.state_dict().items()},
            'target_actor_model': {k: v.cpu() for k, v in self.target_actor_model.state_dict().items()},
            'target_critic_model': {k: v.cpu() for k, v in self.target_critic_model.state_dict().items()},
        }

    def apply_parameters(self, params:Dict[str, Dict[str, T.Tensor]]):
        """Apply params to a model. Used in distributed training."""
        self.actor_model.load_state_dict(params['actor_model'])
        self.critic_model.load_state_dict(params['critic_model'])
        self.target_actor_model.load_state_dict(params['target_actor_model'])
        self.target_critic_model.load_state_dict(params['target_critic_model'])

    def get_action(self, state, goal=None, test=False,
                   state_normalizer:Normalizer=None,
                   goal_normalizer:Normalizer=None):

        # make sure state is a tensor and on correct device
        state = T.tensor(state, dtype=T.float32, device=self.actor_model.device)

        if test:
            if self._use_her:
                state = state_normalizer.normalize(state)
                # make sure goal is a tensor and on correct device
                goal = T.tensor(goal, dtype=T.float32, device=self.actor_model.device)
                goal = goal_normalizer.normalize(goal)
            # use self.state_normalizer if self.normalize_inputs
            # elif self.normalize_inputs and not self._use_her:
            #     state = self.state_normalizer.normalize(state)
            
            with T.no_grad():
                _, action = self.target_actor_model(state, goal)
            return action.cpu().detach().numpy()
                
        # if random number is less than epsilon or in warmup, sample random action
        elif np.random.random() < self.action_epsilon or self._step <= self.warmup:
            action_np = self.env.action_space.sample()
            noise_np = np.zeros((1,action_np.shape[-1]))
        
        else:
            # (HER) use passed state normalizer if using HER
            if self._use_her:
                state = state_normalizer.normalize(state)
                # make sure goal is a tensor and on correct device
                goal = T.tensor(goal, dtype=T.float32, device=self.actor_model.device)
                goal = goal_normalizer.normalize(goal)
            # use self.state_normalizer if self.normalize_inputs
            # elif self.normalize_inputs and not self._use_her:
            #     state = self.state_normalizer.normalize(state)
            
            noise = self.noise()
            if self.noise_schedule:
                noise *= self.noise_schedule.get_factor()
            
            # Switch to eval mode to get action value
            self.actor_model.eval()
            with T.no_grad():
                _, pi = self.actor_model(state, goal)
            self.actor_model.train()

            # Convert the action space bounds to a tensor on the same device
            action_space_high = T.tensor(self.env.action_space.high, dtype=T.float32, device=self.actor_model.device)
            action_space_low = T.tensor(self.env.action_space.low, dtype=T.float32, device=self.actor_model.device)
            action = (pi + noise).clip(action_space_low, action_space_high)

            noise_np = noise.cpu().detach().numpy()
            action_np = action.cpu().detach().numpy()

        # if test:
            # loop over all actions to log to wandb
            # for i, a in enumerate(action_np):
            #     # Log the values to wandb
            #     self._train_step_config[f'action_{i}'] = a

        # Loop over the noise and action values and log them to wandb
        for i in range(action_np.shape[-1]):
            # Log the values to wandb
            # self._train_step_config[f'action_{i}'] = a
            self._train_step_config[f'action_{i}'] = action_np[:,i].mean()
            self._train_step_config[f'action_{i}_noise'] = noise_np[:,i].mean()

        return action_np


    def learn(self, state_normalizer: Normalizer=None, goal_normalizer: Normalizer=None):
        
        self._learn_iter += 1
        self.logger.debug(f"DDPG learn iteration: {self._learn_iter}")
            
        if self.replay_buffer.get_config()['class_name'] == 'PrioritizedReplayBuffer':
            if self._use_her:  # HER with prioritized replay
                #DEBUG
                # print(f"HER with prioritized replay")
                (states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals), weights, probs, indices = self.replay_buffer.sample(self.batch_size)
            else:  # Just prioritized replay
                #DEBUG
                # print(f"Just prioritized replay")
                (states, actions, rewards, next_states, dones), weights, probs, indices = self.replay_buffer.sample(self.batch_size)
                
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
                    
                    
                # Only log metrics if this is the main worker or not using Ray
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        wandb.log({
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
                        }, step=self._step)
        else:  # Standard replay buffer
            if self._use_her:
                #DEBUG
                # print(f"HER with standard replay")
                (states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals) = self.replay_buffer.sample(self.batch_size)
            else:
                #DEBUG
                # print(f"Standard replay")
                (states, actions, rewards, next_states, dones) = self.replay_buffer.sample(self.batch_size)
            
            weights = None
            indices = None

        #DEBUG
        # print(f'states shape: {states.shape}')
        # print(f'states: {states}')
        # print(f'actions shape: {actions.shape}')
        # print(f'actions: {actions}')
        # print(f'rewards shape: {rewards.shape}')
        # print(f'rewards: {rewards}')
        # print(f'next_states shape: {next_states.shape}')
        # print(f'next_states: {next_states}')
        # print(f'dones shape: {dones.shape}')
        # print(f'dones: {dones}')

        # Normalize states if self.normalize_inputs
        if self._use_her:
            states = state_normalizer.normalize(states)
            next_states = state_normalizer.normalize(next_states)
            desired_goals = goal_normalizer.normalize(desired_goals)
        else:
            desired_goals = None

        # Convert rewards and dones to 2D tensors
        # rewards = rewards.unsqueeze(1).to(self.target_critic_model.device)
        # dones = dones.unsqueeze(1).to(self.target_critic_model.device)
        # Don't unsqueeze when using N-step
        rewards = rewards.to(self.target_critic_model.device)
        dones = dones.to(self.target_critic_model.device)

        # Get target values
        with T.no_grad():
            # _, target_actions = self.target_actor_model(next_states, desired_goals)
            _, target_actions = self.target_actor_model(
                next_states[:,-1,:],
                desired_goals[:,-1,:] if desired_goals is not None else None
            ) # N-step
            # target_critic_values = self.target_critic_model(next_states, target_actions, desired_goals)
            # Calculate target Q-values
            # targets = rewards + (1 - dones) * self.discount * target_critic_values
            targets = compute_n_step_return(
                rewards,
                dones,
                self.discount,
                self.N,
                next_states[:,-1,:],
                target_actions,
                desired_goals[:,-1,:] if desired_goals is not None else None,
                self.target_critic_model,
                bootstrap=True,
                device=self.target_critic_model.device
            )
            # Apply HER-specific clamping if needed
            if self._use_her:
                targets = T.clamp(targets, min=-1/(1-self.discount), max=0)
            #DEBUG
            # print(f'targets shape: {targets.shape}')
            # print(f'targets: {targets}')

        # Get current critic predictions
        predictions = self.critic_model(
            states[:,0,:],
            actions[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None
        ).flatten()
        #DEBUG
        # print(f'predictions shape: {predictions.shape}')
        # print(f'predictions: {predictions}')
        # Calculate TD errors
        error = targets - predictions
        #DEBUG
        # print(f'error shape: {error.shape}')
        # print(f'error: {error}')


        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss = (weights.to(self.critic_model.device) * error.pow(2)).mean()
        else:
            critic_loss = error.pow(2).mean()

        # Update critic
        self.critic_model.optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.grad_clip)
        self.critic_model.optimizer.step()

        # Get actor's action predictions
        pre_act_values, action_values = self.actor_model(
            states[:,0,:],
            desired_goals[:,0,:] if desired_goals is not None else None
        )
        
        # Calculate actor loss based on critic
        critic_values = self.critic_model(states[:,0,:], action_values, desired_goals[:,0,:] if desired_goals is not None else None)
        if weights is not None:
            actor_loss = -(weights.to(self.actor_model.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()
        
        # Add HER-specific regularization if needed
        if self._use_her:
            actor_loss += pre_act_values.pow(2).mean()

        # Update actor
        self.actor_model.optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.grad_clip)
        self.actor_model.optimizer.step()

        # Perform soft update on target networks
        if not self._use_her:
            self.soft_update(self.actor_model, self.target_actor_model)
            self.soft_update(self.critic_model, self.target_critic_model)

        # Update priorities if using prioritized replay - only on update_freq steps
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:# and hasattr(self.replay_buffer, 'beta_update_freq'):
            #DEBUG
            # print(f'indices shape: {indices.shape}')
            # print(f'error shape: {error.flatten().shape}')
            self.replay_buffer.update_priorities(indices, error.detach().flatten().to(self.replay_buffer.device))

        # Add metrics to step_logs
        self._train_step_config['td_error'] = error
        self._train_step_config['actor_predictions'] = action_values.mean()
        self._train_step_config['critic_predictions'] = critic_values.mean()
        self._train_step_config['target_actor_predictions'] = target_actions.mean()
        self._train_step_config['target_critic_predictions'] = targets.mean()

        return actor_loss.item() if actor_loss is not None else 0.0, critic_loss.item()
        
    
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

    # @classmethod
    # def sweep_train(
    #     cls,
    #     config, # wandb.config,
    #     train_config,
    #     env_spec,
    #     callbacks,
    #     run_number,
    #     comm=None,
    # ):
    #     """Builds and trains agents from sweep configs. Works with MPI"""
    #     rank = MPI.COMM_WORLD.rank

    #     if comm is not None:
    #         logger.debug(f"Rank {rank} comm detected")
    #         rank = comm.Get_rank()
    #         logger.debug(f"Global rank {MPI.COMM_WORLD.Get_rank()} in {comm.Get_name()} set to comm rank {rank}")
    #         logger.debug(f"init_sweep fired: global rank {MPI.COMM_WORLD.rank}, group rank {rank}, {comm.Get_name()}")
    #     else:
    #         logger.debug(f"init_sweep fired")
    #     try:
    #         # rank = MPI.COMM_WORLD.rank
    #         # Instantiate env from env_spec
    #         env = gym.make(gym.envs.registration.EnvSpec.from_json(env_spec))
    #         # agent_config_path = f'sweep/agent_config_{run_number}.json'
    #         # logger.debug(f"rank {rank} agent config path: {agent_config_path}")
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train config: {train_config}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} env spec id: {env.spec.id}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} callbacks: {callbacks}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} run number: {run_number}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} config set: {config}")
    #         else:
    #             logger.debug(f"train config: {train_config}")
    #             logger.debug(f"env spec id: {env.spec.id}")
    #             logger.debug(f"callbacks: {callbacks}")
    #             logger.debug(f"run number: {run_number}")
    #             logger.debug(f"config set: {config}")
    #         model_type = list(config.keys())[0]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} model type: {model_type}")
    #         else:
    #             logger.debug(f"model type: {model_type}")

    #         actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = wandb_support.format_layers(config)
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} layers built")
    #         else:
    #             logger.debug(f"layers built")
    #         # Actor
    #         actor_learning_rate=config[model_type][f"{model_type}_actor_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor learning rate set")
    #         else:
    #             logger.debug(f"actor learning rate set")
    #         actor_optimizer = config[model_type][f"{model_type}_actor_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer set")
    #         else:
    #             logger.debug(f"actor optimizer set")
    #         # get optimizer params
    #         actor_optimizer_params = {}
    #         if actor_optimizer == "Adam":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            
    #         elif actor_optimizer == "Adagrad":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
            
    #         elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']

    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer params set")
    #         else:
    #             logger.debug(f"actor optimizer params set")
    #         actor_normalize_layers = config[model_type][f"{model_type}_actor_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor normalize layers set")
    #         else:
    #             logger.debug(f"actor normalize layers set")
    #         # Critic
    #         critic_learning_rate=config[model_type][f"{model_type}_critic_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic learning rate set")
    #         else:
    #             logger.debug(f"critic learning rate set")
    #         critic_optimizer = config[model_type][f"{model_type}_critic_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer set")
    #         else:
    #             logger.debug(f"critic optimizer set")
    #         critic_optimizer_params = {}
    #         if critic_optimizer == "Adam":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            
    #         elif critic_optimizer == "Adagrad":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
            
    #         elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer params set")
    #         else:
    #             logger.debug(f"critic optimizer params set")

    #         critic_normalize_layers = config[model_type][f"{model_type}_critic_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic normalize layers set")
    #         else:
    #             logger.debug(f"critic normalize layers set")
    #         # Set device
    #         device = config[model_type][f"{model_type}_device"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} device set")
    #         else:
    #             logger.debug(f"device set")
    #         # Check if CNN layers and if so, build CNN model
    #         if actor_cnn_layers:
    #             actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
    #         else:
    #             actor_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor cnn layers set: {actor_cnn_layers}")
    #         else:
    #             logger.debug(f"actor cnn layers set: {actor_cnn_layers}")

    #         if critic_cnn_layers:
    #             critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
    #         else:
    #             critic_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic cnn layers set: {critic_cnn_layers}")
    #         else:
    #             logger.debug(f"critic cnn layers set: {critic_cnn_layers}")
    #         # Get actor clamp value
    #         # clamp_output = config[model_type][f"{model_type}_actor_clamp_output"]
    #         # if comm is not None:
    #         #     logger.debug(f"{comm.Get_name()}; Rank {rank} clamp output set: {clamp_output}")
    #         # else:
    #         #     logger.debug(f"clamp output set: {clamp_output}")
    #         actor_model = models.ActorModel(env = env,
    #                                         cnn_model = actor_cnn_model,
    #                                         dense_layers = actor_layers,
    #                                         output_layer_kernel=kernels[f'actor_output_kernel'],
    #                                         optimizer = actor_optimizer,
    #                                         optimizer_params = actor_optimizer_params,
    #                                         learning_rate = actor_learning_rate,
    #                                         normalize_layers = actor_normalize_layers,
    #                                         # clamp_output=clamp_output,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor model built: {actor_model.get_config()}")
    #         else:
    #             logger.debug(f"actor model built: {actor_model.get_config()}")
    #         critic_model = models.CriticModel(env = env,
    #                                         cnn_model = critic_cnn_model,
    #                                         state_layers = critic_state_layers,
    #                                         merged_layers = critic_merged_layers,
    #                                         output_layer_kernel=kernels[f'critic_output_kernel'],
    #                                         optimizer = critic_optimizer,
    #                                         optimizer_params = critic_optimizer_params,
    #                                         learning_rate = critic_learning_rate,
    #                                         normalize_layers = critic_normalize_layers,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic model built: {critic_model.get_config()}")
    #         else:
    #             logger.debug(f"critic model built: {critic_model.get_config()}")
    #         # get normalizer clip value
    #         normalizer_clip = config[model_type][f"{model_type}_normalizer_clip"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} normalizer clip set: {normalizer_clip}")
    #         else:
    #             logger.debug(f"normalizer clip set: {normalizer_clip}")
    #         # get action epsilon
    #         action_epsilon = config[model_type][f"{model_type}_epsilon_greedy"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} action epsilon set: {action_epsilon}")
    #         else:
    #             logger.debug(f"action epsilon set: {action_epsilon}")
    #         # Replay buffer size
    #         replay_buffer_size = config[model_type][f"{model_type}_replay_buffer_size"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} replay buffer size set: {replay_buffer_size}")
    #         else:
    #             logger.debug(f"replay buffer size set: {replay_buffer_size}")
    #         # Save dir
    #         save_dir = config[model_type][f"{model_type}_save_dir"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} save dir set: {save_dir}")
    #         else:
    #             logger.debug(f"save dir set: {save_dir}")

    #         # create replay buffer
    #         replay_buffer = ReplayBuffer(env, replay_buffer_size, device=device)
    #         # create DDPG agent
    #         ddpg_agent= cls(
    #             env = env,
    #             actor_model = actor_model,
    #             critic_model = critic_model,
    #             discount = config[model_type][f"{model_type}_discount"],
    #             tau = config[model_type][f"{model_type}_tau"],
    #             action_epsilon = action_epsilon,
    #             replay_buffer = replay_buffer,
    #             batch_size = config[model_type][f"{model_type}_batch_size"],
    #             noise = Noise.create_instance(config[model_type][f"{model_type}_noise"], shape=env.action_space.shape, **config[model_type][f"{model_type}_noise_{config[model_type][f'{model_type}_noise']}"], device=device),
    #             warmup = config[model_type][f"{model_type}_warmup"],
    #             callbacks = callbacks,
    #             comm = comm,
    #             device = device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} ddpg agent built: {ddpg_agent.get_config()}")
    #         else:
    #             logger.debug(f"ddpg agent built: {ddpg_agent.get_config()}")

    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier called")
    #         else:
    #             logger.debug(f"train barrier called")

    #         if comm is not None:
    #             comm.Barrier()
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier passed")

    #         ddpg_agent.train(
    #                 num_episodes=train_config['num_episodes'],
    #                 render=False,
    #                 render_freq=0,
    #                 )

    #     except Exception as e:
    #         logger.error(f"An error occurred: {e}", exc_info=True)

    def train(self, num_episodes: int, num_envs: int, seed: int | None = None, render_freq: int = 0, sync_iter: int = 1):
        """Trains the model for 'episodes' number of episodes."""

        # set models to train mode
        self.actor_model.train()
        self.critic_model.train()
        # Set target models to eval mode
        self.target_actor_model.eval()
        self.target_critic_model.eval()

         # set num_envs as attribute
        self.num_envs = num_envs

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        # if render_freq == None:
        #     render_freq = 0

        # Set seeds
        set_seed(seed)

        # Set sync_interval (for distributed learning)
        self._sync_iter = sync_iter

        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, WandbCallback):
                    config = self.get_config()
                    config['num_episodes'] = num_episodes
                    config['seed'] = seed
                    config['num_envs'] = self.num_envs
                    config['distributed'] = self._distributed
                    config['sync_interval'] = self._sync_iter
                    callback.on_train_begin((self.actor_model, self.critic_model,), logs=config)
                    run_number = callback.run_name.split("-")[-1]
        
        try:
            # instantiate new vec environment
            self.env.env = self.env._initialize_env(0, self.num_envs, seed)
        except Exception as e:
            self.logger.error(f"Error in DDPG.train self.env")
        
        # initialize step counter (for logging)
        self._step = 0
        best_reward = -np.inf
        score_history = deque(maxlen=100)
        # trajectories = [[] for _ in range(self.num_envs)]
        episode_scores = np.zeros(self.num_envs)
        self.completed_episodes = np.zeros(self.num_envs)
        # Initialize environments
        states, _ = self.env.reset()
        while self.completed_episodes.sum() < num_episodes:
            # If distributed, sync to shared agent
            if self._distributed and self._step % self._sync_iter == 0:
                params = self.get_parameters()
                self.apply_parameters(params)
            self._step += 1

            rendered = False # Flag to keep track of render status to avoid rendering multiple times per step
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            # reset noise
            if type(self.noise) == OUNoise:
                self.noise.reset()
            actions = self.get_action(states)
            # Format actions
            actions = self.env.format_actions(actions)
            next_states, rewards, dones, _, traj_ids, step_indices = self.env.step(actions)
            episode_scores += rewards
            # dones = np.logical_or(terms, truncs)
            
            # Store transitions in the env trajectory
            # for i in range(self.num_envs):
            #     self.replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i], traj_ids[i], step_indices[i])
                # trajectories[i].append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
            self.replay_buffer.add(states, actions, rewards, next_states, dones, traj_ids, step_indices)

            completed_episodes = np.flatnonzero(dones) # Get indices of completed episodes
            for i in completed_episodes:
                # self.replay_buffer.add(*zip(*trajectories[i]))
                # trajectories[i] = []

                # Increment completed episodes for env by 1
                self.completed_episodes[i] += 1
                score_history.append(episode_scores[i]) 
                avg_reward = sum(score_history) / len(score_history)
                self._train_episode_config['episode'] = self.completed_episodes.sum()
                self._train_episode_config["episode_reward"] = episode_scores[i]

                # check if best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self._train_episode_config["best"] = 1
                    # save model
                    self.save()
                else:
                    self._train_episode_config["best"] = 0

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

                # Check if number of completed episodes should trigger render
                if self.completed_episodes.sum() % render_freq == 0 and not rendered:
                    print(f"Rendering episode {self.completed_episodes.sum()} during training...")
                    # Call the test function to render an episode
                    self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
                    # Add render to wandb log
                    video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes.sum()}.mp4")
                    # Log the video to wandb
                    if self.callbacks:
                        for callback in self.callbacks:
                            if isinstance(callback, WandbCallback):
                                # Only log videos if this is the main worker or not using Ray
                                if not hasattr(callback, 'is_main_worker') or callback.is_main_worker:
                                    wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")}, step=self._step)
                    rendered = True
                    # Switch models back to train mode after rendering
                    # self.actor_model.train()
                    # self.critic_model.train()
                # else:
                #     rendered = False

                print(f"Environment {i}: Episode {int(self.completed_episodes.sum())}, Score {episode_scores[i]}, Avg_Score {avg_reward}")

                # Reset score of episode to 0
                episode_scores[i] = 0
                    
            states = next_states
            
            # Check if past warmup
            if self._step > self.warmup:
                # check if enough samples in replay buffer and if so, learn from experiences
                if self.replay_buffer.counter > self.batch_size:
                    # Check if distributed
                    if self._distributed:
                        self._distributed_learn(self._step, run_number)
                    else:
                        actor_loss, critic_loss = self.learn()
                        self._train_step_config["actor_loss"] = actor_loss
                        self._train_step_config["critic_loss"] = critic_loss
                    # Step scheduler if not None
                    if self.noise_schedule:
                        self.noise_schedule.step()
                        self._train_step_config["noise_anneal"] = self.noise_schedule.get_factor()


            self._train_step_config["step_reward"] = rewards.mean()
            
            # log to wandb if using wandb callback
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_end(step=self._step, logs=self._train_step_config)

            

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)
       
    def test(self, num_episodes: int, num_envs: int=1, seed: int=None, render_freq: int=0, training: bool=False):
        """Runs a test over 'num_episodes'."""

        # set model in eval mode
        # self.actor_model.eval()
        # self.critic_model.eval()

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        if render_freq == None:
            render_freq = 0

        # Set seeds
        set_seed(seed)

        try:
            # instantiate new vec environment
            # env = self.env._initialize_env(render_freq, num_envs, seed)
            env = EnvWrapper.from_json(self.env.to_json())
            env.env = env._initialize_env(render_freq, 1, seed)
        except Exception as e:
            self.logger.error(f"Error in ddpg.test agent._initialize_env process: {e}", exc_info=True)

        if self.callbacks and not training:
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    # Add to config to send to wandb for logging
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)

        _step = 0
        # Instantiate array to keep track of number of completed episodes per env
        completed_episodes = np.zeros(num_envs)
        # Instantiate array to keep track of current episode scores
        episode_scores = np.zeros(num_envs)
        # Instantiate a deque to track last 'episodes_per_update' scores for computing avg
        completed_scores = deque(maxlen=num_episodes)
        # Instantiate list to keep track of frames for rendering
        frames = []
        # Reset environment to get starting state
        states, _ = env.reset()
        while completed_episodes.sum() < num_episodes:
            # Increment step counter
            _step += 1
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_begin(epoch=_step, logs=None)
            
            # if self.callbacks:
            #     for callback in self.callbacks:
            #         callback.on_train_step_begin(step=self._step, logs=None)
            actions = self.get_action(states, test=True)
            # Format actions
            actions = self.env.format_actions(actions, testing=True)
            next_states, rewards, dones, _ = env.step(actions, testing=True)
            self._test_step_config["step_reward"] = rewards
            episode_scores += rewards
            # dones = np.logical_or(terms, truncs)

            if render_freq > 0:
                # Capture the frame
                frame = env.env.render()[0]
                # print(f'frame:{frame}')
                frames.append(frame)

            for i in range(num_envs):
                if dones[i]:
                    # Increment completed episodes for env by 1
                    completed_episodes[i] += 1
                    # Append environment score to completed scores
                    completed_scores.append(episode_scores[i])
                    # Add the episode reward to the episode log for callbacks
                    self._test_episode_config["episode_reward"] = episode_scores[i]
                    # Reset the episode score of the env back to 0
                    episode_scores[i] = 0
                    # check if best reward
                    avg_reward = sum(completed_scores) / len(completed_scores)
                    # Log completed episodes to callback episode config
                    self._test_episode_config["episode"] = completed_episodes.sum()
                    # Save the video if the episode number is divisible by render_freq
                    if (render_freq > 0) and ((completed_episodes.sum()) % render_freq == 0):
                        if training:
                            render_video(frames, self.completed_episodes.sum(), self.save_dir, 'train')
                        else:
                            render_video(frames, completed_episodes.sum(), self.save_dir, 'test')
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/test/episode_{completed_episodes.sum()}.mp4")
                            # Log the video to wandb
                            if self.callbacks:
                                for callback in self.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        # Only log videos if this is the main worker or not using Ray
                                        if not hasattr(callback, 'is_main_worker') or callback.is_main_worker:
                                            wandb.log({"testing_video": wandb.Video(video_path, caption="Testing process", format="mp4")})
                        # Empty frames array
                        frames = []
                    # Signal to all callbacks that an episode (epoch) has completed and to log data
                    if self.callbacks and not training:
                        for callback in self.callbacks:
                            callback.on_test_epoch_end(
                            epoch=_step, logs=self._test_episode_config
                        )
                    if not training:
                        # Print episode update to console
                        print(f"Environment {i}: Episode {int(completed_episodes.sum())}/{num_episodes} Score: {completed_scores[-1]} Avg Score: {avg_reward}")

            states = next_states

            # Perform an update if the number of completed trajectories is greater than or
            # equal to the number of trajectories per update
            # if len(completed_trajectories) >= trajectories_per_update:
            #     self.learn(completed_trajectories)
            #     # Clear completed_trajectories
            #     completed_trajectories = []

            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=_step, logs=self._test_step_config)

        if self.callbacks and not training:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)


    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.to_json(),
            "actor_model": self.actor_model.get_config(),
            "critic_model": self.critic_model.get_config(),
            "replay_buffer": self.replay_buffer.get_config() if self.replay_buffer is not None else None,
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "batch_size": self.batch_size,
            "noise": self.noise.get_config(),
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            'grad_clip': self.grad_clip,
            'warmup': self.warmup,
            'N': self.N,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir,
            "device": self.device.type,
            "log_level": logging.getLevelName(self.logger.getEffectiveLevel()).lower()
        }


    def save(self):
        """Saves the model."""

        # Change self.save_dir if save_dir
        # if save_dir is not None:
        #     self.save_dir = save_dir + "/ddpg/"

        config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of DDPG agent config
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        # saves policy and value model
        self.actor_model.save(self.save_dir)
        self.critic_model.save(self.save_dir)
        
        # if self.normalize_inputs:
        #     self.state_normalizer.save_state(self.save_dir + "state_normalizer.npz")

    @classmethod
    def load(cls, config, load_weights=True):
        """Loads the model."""

        # Load EnvWrapper
        env_wrapper = EnvWrapper.from_json(config["env"])
            
        # load policy model
        actor_model = ActorModel.load(config['actor_model'], load_weights)
        # load value model
        critic_model = CriticModel.load(config['critic_model'], load_weights)
        # load replay buffer if not None
        if config['replay_buffer'] is not None:
            config['replay_buffer']['config']['env'] = env_wrapper
            if config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer':
                replay_buffer = PrioritizedReplayBuffer(**config["replay_buffer"]["config"])
            else:
                replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
        else:
            replay_buffer = None
        # load noise
        noise = Noise.create_instance(config["noise"]["class_name"], **config["noise"]["config"])
        # load callbacks
        callbacks = [callback_load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]\
                    if config['callbacks'] else None

        # return DDPG agent
        agent = cls(
            env = env_wrapper,
            actor_model = actor_model,
            critic_model = critic_model,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            replay_buffer=replay_buffer,
            batch_size=config["batch_size"],
            noise=noise,
            noise_schedule=ScheduleWrapper(config["noise_schedule"]),
            grad_clip=config['grad_clip'],
            warmup = config['warmup'],
            N = config['N'],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
            log_level=config["log_level"]
        )

        # if agent.normalize_inputs:
        #     agent.state_normalizer = Normalizer.load_state(config['save_dir'] + "state_normalizer.npz")

        return agent
    

class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""
    
    def __init__(
        self,
        env: EnvWrapper,
        actor_model: ActorModel,
        critic_model_a: CriticModel,
        critic_model_b: Optional[CriticModel] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        action_epsilon: float = 0.0,
        replay_buffer: Buffer = None,
        batch_size: int = 256,
        noise: Noise = None,
        noise_schedule: ScheduleWrapper=None,
        target_noise: Noise = None,
        target_noise_schedule: ScheduleWrapper=None,
        target_noise_clip: float = 0.5,
        actor_update_delay: int = 2,
        grad_clip: float = 40.0,
        warmup: int = 1000,
        N: int=1, # N-steps
        callbacks: list = None,
        save_dir: str = "models",
        device: str = None,
        log_level: str = 'info'
    ):
        try:
            super().__init__(env, callbacks, save_dir, device, log_level)
            self.actor_model = actor_model
            self.critic_model_a = critic_model_a
            self.critic_model_b = critic_model_b
            # clone second critic (do not copy weights) if critic_model_b None
            if not critic_model_b:
                self.critic_model_b = self.clone_model(self.critic_model_a, copy_weights=False)
            # set target networks as clones of the main networks
            self.target_actor_model = self.clone_model(self.actor_model)
            self.target_critic_model_a = self.clone_model(self.critic_model_a)
            self.target_critic_model_b = self.clone_model(self.critic_model_b)
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
            self.target_noise_clip = target_noise_clip
            self.actor_update_delay = actor_update_delay
            self.grad_clip = grad_clip
            self.warmup = warmup
            self.N = N

        except Exception as e:
            self.logger.error(f"Error in TD3 init: {e}", exc_info=True)
        
        try:
            # Determine the observation shape
            obs_space = (self.env.single_observation_space 
                         if hasattr(self.env, "single_observation_space") 
                         else self.env.observation_space)
            if isinstance(obs_space, gym.spaces.Dict):
                self._obs_space_shape = obs_space['observation'].shape
            else:
                self._obs_space_shape = obs_space.shape
            # if self.normalize_inputs:
            #     self.state_normalizer = Normalizer(self._obs_space_shape, self.normalizer_eps, self.normalizer_clip, device=device)
        except Exception as e:
            self.logger.error(f"Error in TD3 init internal attributes: {e}", exc_info=True)

        # instantiate internal attribute use_her to be switched by HER class if using TD3
        self._use_her = False
        self._opt_step = 0 # number of times optimizer (learn()) has run

        # Set learn iter to 0. For distributed training
        self._learn_iter = 0

    def clone_model(self, model, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Clones a model."""
        if device:
            device = get_device(device)
        else:
            device = self.device

        return model.clone(copy_weights, device)
    
    def _initialize_wandb(self, run_number:str=None, run_name_prefix:str=None):
        """Initialize WandbCallback if using WandbCallback"""
        try:
            if self._wandb:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if not callback.initialized:
                            models = (self.actor_model, self.critic_model_a, self.critic_model_b)
                            config = self.get_config()
                            callback.initialize_run(models, config, run_number=run_number, run_name_prefix=run_name_prefix)
        except Exception as e:
            self.logger.error(f"Error in _initialize_wandb: {e}", exc_info=True)

    def _distributed_learn(self, step: int, run_number:str=None):
        """Used in distributed training to update the shared models.
        This function is overridden by the Worker class to point to the Learner class.
        """
        self.logger.debug(f"TD3 distributed learn iteration: {step}")
        previous_step = self._step
        # Set current step to step if greater than current step
        if step > previous_step:
            self._step = step
            # Initialize wandb check
            self._initialize_wandb(run_number=run_number, run_name_prefix="train")
            actor_loss, critic_loss = self.learn()
            # Only store log if current step greater than previous and self._wandb
            if self._wandb:
                self._train_step_config["actor_loss"] = actor_loss
                self._train_step_config["critic_loss"] = critic_loss
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        callback.on_train_step_end(step, self._train_step_config)
        else:
            actor_loss, critic_loss = self.learn()

    def get_parameters(self):
        """Get the parameters of all models."""
        return {
            'actor_model': [param.data.clone().to('cpu') for param in self.actor_model.parameters()],
            'critic_model_a': [param.data.clone().to('cpu') for param in self.critic_model_a.parameters()],
            'critic_model_b': [param.data.clone().to('cpu') for param in self.critic_model_b.parameters()],
            'target_actor_model': [param.data.clone().to('cpu') for param in self.target_actor_model.parameters()],
            'target_critic_model_a': [param.data.clone().to('cpu') for param in self.target_critic_model_a.parameters()],
            'target_critic_model_b': [param.data.clone().to('cpu') for param in self.target_critic_model_b.parameters()],
        }

    def apply_parameters(self, params:Dict[str, List[T.Tensor]]):
        """Apply params to a model. Used in distributed training."""
        for param, new_param in zip(self.actor_model.parameters(), params['actor_model']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
        for param, new_param in zip(self.critic_model_a.parameters(), params['critic_model_a']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
        for param, new_param in zip(self.critic_model_b.parameters(), params['critic_model_b']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
        for param, new_param in zip(self.target_actor_model.parameters(), params['target_actor_model']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
        for param, new_param in zip(self.target_critic_model_a.parameters(), params['target_critic_model_a']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
        for param, new_param in zip(self.target_critic_model_b.parameters(), params['target_critic_model_b']):
            if new_param is not None:
                param.data.copy_(new_param.to('cpu'))
    
    # @classmethod
    # def build(
    #     cls,
    #     env,
    #     actor_cnn_layers,
    #     critic_cnn_layers,
    #     actor_layers,
    #     critic_state_layers,
    #     critic_merged_layers,
    #     kernels,
    #     callbacks,
    #     config,#: wandb.config,
    #     save_dir: str = "models/",
    # ):
    #     """Builds the agent."""
    #     # Actor
    #     actor_learning_rate=config[config.model_type][f"{config.model_type}_actor_learning_rate"]
    #     actor_optimizer = config[config.model_type][f"{config.model_type}_actor_optimizer"]
    #     # get optimizer params
    #     actor_optimizer_params = {}
    #     if actor_optimizer == "Adam":
    #         actor_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
        
    #     elif actor_optimizer == "Adagrad":
    #         actor_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #         actor_optimizer_params['lr_decay'] = \
    #             config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
        
    #     elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
    #         actor_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #         actor_optimizer_params['momentum'] = \
    #             config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']

    #     actor_normalize_layers = config[config.model_type][f"{config.model_type}_actor_normalize_layers"]

    #     # Critic
    #     critic_learning_rate=config[config.model_type][f"{config.model_type}_critic_learning_rate"]
    #     critic_optimizer = config[config.model_type][f"{config.model_type}_critic_optimizer"]
    #     critic_optimizer_params = {}
    #     if critic_optimizer == "Adam":
    #         critic_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
        
    #     elif critic_optimizer == "Adagrad":
    #         critic_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #         critic_optimizer_params['lr_decay'] = \
    #             config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
        
    #     elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
    #         critic_optimizer_params['weight_decay'] = \
    #             config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #         critic_optimizer_params['momentum'] = \
    #             config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
        
    #     critic_normalize_layers = config[config.model_type][f"{config.model_type}_critic_normalize_layers"]

    #     # Check if CNN layers and if so, build CNN model
    #     if actor_cnn_layers:
    #         actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
    #     else:
    #         actor_cnn_model = None

    #     if critic_cnn_layers:
    #         critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
    #     else:
    #         critic_cnn_model = None

    #     # Set device
    #     device = config[config.model_type][f"{config.model_type}_device"]

    #     # get desired, achieved, reward func for env
    #     desired_goal_func, achieved_goal_func, reward_func = gym_helper.get_her_goal_functions(env)
    #     goal_shape = desired_goal_func(env).shape

    #     # Get actor clamp value
    #     # clamp_output = config[config.model_type][f"{config.model_type}_actor_clamp_output"]
        
    #     actor_model = models.ActorModel(env = env,
    #                                     cnn_model = actor_cnn_model,
    #                                     dense_layers = actor_layers,
    #                                     output_layer_kernel=kernels[f'actor_output_kernel'],
    #                                     goal_shape=goal_shape,
    #                                     optimizer = actor_optimizer,
    #                                     optimizer_params = actor_optimizer_params,
    #                                     learning_rate = actor_learning_rate,
    #                                     normalize_layers = actor_normalize_layers,
    #                                     # clamp_output=clamp_output,
    #                                     device=device,
    #     )
    #     critic_model = models.CriticModel(env = env,
    #                                       cnn_model = critic_cnn_model,
    #                                       state_layers = critic_state_layers,
    #                                       merged_layers = critic_merged_layers,
    #                                       output_layer_kernel=kernels[f'critic_output_kernel'],
    #                                       goal_shape=goal_shape,
    #                                       optimizer = critic_optimizer,
    #                                       optimizer_params = critic_optimizer_params,
    #                                       learning_rate = critic_learning_rate,
    #                                       normalize_layers = critic_normalize_layers,
    #                                       device=device,
    #     )

    #     # action epsilon
    #     action_epsilon = config[config.model_type][f"{config.model_type}_epsilon_greedy"]

    #     # normalize inputs
    #     normalize_inputs = config[config.model_type][f"{config.model_type}_normalize_input"]
    #     # normalize_kwargs = {}
    #     if "True" in normalize_inputs:
    #         # normalize_kwargs = config[config.model_type][f"{config.model_type}_normalize_clip"]
    #         normalizer_clip = config[config.model_type][f"{config.model_type}_normalize_clip"]

    #     agent = cls(
    #         env = env,
    #         actor_model = actor_model,
    #         critic_model = critic_model,
    #         discount = config[config.model_type][f"{config.model_type}_discount"],
    #         tau = config[config.model_type][f"{config.model_type}_tau"],
    #         action_epsilon = action_epsilon,
    #         replay_buffer = ReplayBuffer(env=env),
    #         batch_size = config[config.model_type][f"{config.model_type}_batch_size"],
    #         noise = Noise.create_instance(config[config.model_type][f"{config.model_type}_noise"], shape=env.action_space.shape, **config[config.model_type][f"{config.model_type}_noise_{config[config.model_type][f'{config.model_type}_noise']}"]),
    #         normalize_inputs = normalize_inputs,
    #         # normalize_kwargs = normalize_kwargs,
    #         normalizer_clip = normalizer_clip,
    #         callbacks = callbacks,
    #         save_dir = save_dir,
    #     )

    #     agent.save()

    #     return agent
    
    def _init_her(self):
            self._use_her = True

    def get_action(self, state, goal=None, test=False,
                   state_normalizer:Normalizer=None,
                   goal_normalizer:Normalizer=None):

        # make sure state is a tensor and on correct device
        state = T.tensor(state, dtype=T.float32, device=self.actor_model.device)
        if goal is not None:
            goal = T.tensor(goal, dtype=T.float32, device=self.actor_model.device)
        
        # check if get action is for testing
        if test:
            with T.no_grad():
                # (HER) else if using HER, normalize using passed normalizer
                if self._use_her:
                    state = state_normalizer.normalize(state)
                    goal = goal_normalizer.normalize(goal)

                with T.no_grad():
                    _, action = self.target_actor_model(state, goal)
                # transfer action to cpu, detach from any graphs, tranform to numpy, and flatten
                action_np = action.cpu().detach().numpy()#.flatten()
        
        else:
            # check if using epsilon greedy
            if np.random.random() < self.action_epsilon or self._step <= self.warmup:
                action_np = self.env.action_space.sample()
                noise_np = np.zeros((1,action_np.shape[-1]))
            
            else:
                if self._use_her:
                    state = state_normalizer.normalize(state)
                    # make sure goal is a tensor and on correct device
                    goal = T.tensor(goal, dtype=T.float32, device=self.actor_model.device)
                    goal = goal_normalizer.normalize(goal)
                
                # Create noise
                noise = self.noise()
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()

                # Switch to eval mode to get action value
                self.actor_model.eval()
                with T.no_grad():
                    _, pi = self.actor_model(state, goal)
                self.actor_model.train()
                # print(f'pi: {pi}')

                # Convert the action space bounds to a tensor on the same device
                action_space_high = T.tensor(self.env.single_action_space.high, dtype=T.float32, device=self.actor_model.device)
                action_space_low = T.tensor(self.env.single_action_space.low, dtype=T.float32, device=self.actor_model.device)
                action = (pi + noise).clip(action_space_low, action_space_high)
                # print(f'action + noise: {action}')

                noise_np = noise.cpu().detach().numpy()#.flatten()
                action_np = action.cpu().detach().numpy()#.flatten()
                # print(f'action np: {action_np}')

        if test:
            # loop over all actions to log to wandb
            # for i, a in enumerate(action_np):
            #     # Log the values to wandb
            #     self._train_step_config[f'action_{i}'] = a
            for i in range(action_np.shape[-1]):
                # Log the values to wandb
                self._test_step_config[f'action_{i}'] = action_np[:,i].mean()
                # self._train_step_config[f'action_{i}_noise'] = noise_np[i]

        else:
            # Loop over the noise and action values and log them to wandb
            # for i, (a,n) in enumerate(zip(action_np, noise_np)):
            #     # Log the values to wandb
            #     self._train_step_config[f'action_{i}'] = a
            #     self._train_step_config[f'noise_{i}'] = n

            # Loop over the noise and action values and log them to wandb
            for i in range(action_np.shape[-1]):
                # Log the values to wandb
                # self._train_step_config[f'action_{i}'] = a
                self._train_step_config[f'action_{i}'] = action_np[:,i].mean()
                self._train_step_config[f'action_{i}_noise'] = noise_np[:,i].mean()
        
        # print(f'pi: {pi}; noise: {noise}; action_np: {action_np}')

        return action_np


    def learn(self, state_normalizer: Normalizer = None, goal_normalizer: Normalizer = None):
        self._opt_step += 1

        if self.replay_buffer.get_config()['class_name'] == 'PrioritizedReplayBuffer':
            if self._use_her:  # HER with prioritized replay
                #DEBUG
                # print(f"HER with prioritized replay")
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals, weights, probs, indices = self.replay_buffer.sample(self.batch_size)
            else:  # Just prioritized replay
                #DEBUG
                # print(f"Just prioritized replay")
                states, actions, rewards, next_states, dones, weights, probs, indices = self.replay_buffer.sample(self.batch_size)

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
                    
                    
                # Only log metrics if this is the main worker or not using Ray
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        wandb.log({
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
                        }, step=self._step)
        else:  # Standard replay buffer
            if self._use_her:
                #DEBUG
                # print(f"HER with standard replay")
                states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals = self.replay_buffer.sample(self.batch_size)
            else:
                #DEBUG
                # print(f"Standard replay")
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            weights = None
            indices = None

        # Normalize states if self.normalize_inputs
        if self._use_her:
            states = state_normalizer.normalize(states)
            next_states = state_normalizer.normalize(next_states)
            desired_goals = goal_normalizer.normalize(desired_goals)
        else:

            desired_goals = None
       # Convert rewards and dones to 2D tensors
        rewards = rewards.unsqueeze(1).to(self.target_critic_model_a.device)
        dones = dones.unsqueeze(1).to(self.target_critic_model_a.device)

        # Get target values
        with T.no_grad():
            _, target_actions = self.target_actor_model(next_states, desired_goals)
            noise = self.target_noise()
            
            # Apply noise clipping if needed
            if self.target_noise_clip > 0:
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                
            # Apply noise scaling if scheduled
            if self.target_noise_schedule is not None:
                noise *= self.target_noise_schedule.get_factor()
                
            # Add noise to target actions and clamp to action space
            target_actions = (target_actions + noise).clamp(float(self.env.action_space.low.min()), float(self.env.action_space.high.max()))
            
            # Get target critic values from both critic networks
            target_critic_values_a = self.target_critic_model_a(next_states, target_actions, desired_goals)
            target_critic_values_b = self.target_critic_model_b(next_states, target_actions, desired_goals)
            
            # Take minimum of both critic values for stability
            target_critic_values = T.min(target_critic_values_a, target_critic_values_b)
            
            # Calculate target Q-values
            targets = rewards + (1 - dones) * self.discount * target_critic_values
            
            # Apply HER-specific clamping if needed
            if self._use_her:
                targets = T.clamp(targets, min=-1/(1-self.discount), max=0)

        # Get current critic predictions
        predictions_a = self.critic_model_a(states, actions, desired_goals)
        predictions_b = self.critic_model_b(states, actions, desired_goals)

        # Calculate TD errors (use average of both critic networks for PER)
        error_a = targets - predictions_a
        error_b = targets - predictions_b
        error = (error_a.abs() + error_b.abs()) / 2  # Average of absolute errors for priorities

        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            critic_loss_a = (weights.to(self.critic_model_a.device) * error_a.pow(2)).mean()
            critic_loss_b = (weights.to(self.critic_model_b.device) * error_b.pow(2)).mean()
            critic_loss = critic_loss_a + critic_loss_b
        else:
            critic_loss = F.mse_loss(predictions_a, targets) + F.mse_loss(predictions_b, targets)

        # Update critics
        self.critic_model_a.optimizer.zero_grad()
        self.critic_model_b.optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            T.nn.utils.clip_grad_norm_(self.critic_model_a.parameters(), self.grad_clip)
            T.nn.utils.clip_grad_norm_(self.critic_model_b.parameters(), self.grad_clip)
        self.critic_model_a.optimizer.step()
        self.critic_model_b.optimizer.step()
        
        # Get actor's action predictions
        pre_act_values, action_values = self.actor_model(states, desired_goals)
        
        # Calculate actor loss based on critic A
        critic_values = self.critic_model_a(states, action_values, desired_goals)
        actor_loss = -T.mean(critic_values)
        
        # Add HER-specific regularization if using HER
        if self._use_her:
            actor_loss += pre_act_values.pow(2).mean()

        
        # Update actor
        # Only update actor every actor_update_delay steps
        if self._opt_step % self.actor_update_delay == 0:
            self.actor_model.optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                T.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.grad_clip)
            self.actor_model.optimizer.step()

            if not self._use_her:
                # Perform soft update on target networks
                self.soft_update(self.actor_model, self.target_actor_model)
                self.soft_update(self.critic_model_a, self.target_critic_model_a)
                self.soft_update(self.critic_model_b, self.target_critic_model_b)
        # else:
        #     actor_loss = None
        #     action_values = actions  # Use original actions for metrics

        # Update priorities if using prioritized replay - only on update_freq steps
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:
            self.replay_buffer.update_priorities(indices, error.detach().flatten().to(self.replay_buffer.device))

        # Add metrics to step_logs
        self._train_step_config['actor_predictions'] = action_values.mean()
        self._train_step_config['critic_predictions'] = critic_values.mean() if 'critic_values' in locals() else predictions_a.mean()
        self._train_step_config['target_actor_predictions'] = target_actions.mean()
        self._train_step_config['target_critic_predictions'] = target_critic_values.mean()
        self._train_step_config['target_noise'] = noise.mean()

        return actor_loss.item(), critic_loss.item()
        
    
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
        
    # @classmethod
    # def sweep_train(
    #     cls,
    #     config, # wandb.config,
    #     train_config,
    #     env_spec,
    #     callbacks,
    #     run_number,
    #     comm=None,
    # ):
    #     """Builds and trains agents from sweep configs. Works with MPI"""
    #     rank = MPI.COMM_WORLD.rank

    #     if comm is not None:
    #         logger.debug(f"Rank {rank} comm detected")
    #         rank = comm.Get_rank()
    #         logger.debug(f"Global rank {MPI.COMM_WORLD.Get_rank()} in {comm.Get_name()} set to comm rank {rank}")
    #         logger.debug(f"init_sweep fired: global rank {MPI.COMM_WORLD.rank}, group rank {rank}, {comm.Get_name()}")
    #     else:
    #         logger.debug(f"init_sweep fired: global rank")
    #     try:
    #         # rank = MPI.COMM_WORLD.rank
    #         # Instantiate env from env_spec
    #         env = gym.make(gym.envs.registration.EnvSpec.from_json(env_spec))
    #         # agent_config_path = f'sweep/agent_config_{run_number}.json'
    #         # logger.debug(f"rank {rank} agent config path: {agent_config_path}")
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train config: {train_config}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} env spec id: {env.spec.id}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} callbacks: {callbacks}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} run number: {run_number}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} config set: {config}")
    #         else:
    #             logger.debug(f"train config: {train_config}")
    #             logger.debug(f"env spec id: {env.spec.id}")
    #             logger.debug(f"callbacks: {callbacks}")
    #             logger.debug(f"run number: {run_number}")
    #             logger.debug(f"config set: {config}")
    #         model_type = list(config.keys())[0]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} model type: {model_type}")
    #         else:
    #             logger.debug(f"model type: {model_type}")
    #         # Only primary process (rank 0) calls wandb.init() to build agent and log data

    #         actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = wandb_support.format_layers(config)
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} layers built")
    #         else:
    #             logger.debug(f"layers built")
    #         # Actor
    #         actor_learning_rate=config[model_type][f"{model_type}_actor_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor learning rate set")
    #         else:
    #             logger.debug(f"actor learning rate set")
    #         actor_optimizer = config[model_type][f"{model_type}_actor_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer set")
    #         else:
    #             logger.debug(f"actor optimizer set")
    #         # get optimizer params
    #         actor_optimizer_params = {}
    #         if actor_optimizer == "Adam":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            
    #         elif actor_optimizer == "Adagrad":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
            
    #         elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer params set")
    #         else:
    #             logger.debug(f"actor optimizer params set")
    #         actor_normalize_layers = config[model_type][f"{model_type}_actor_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor normalize layers set")
    #         else:
    #             logger.debug(f"actor normalize layers set")
    #         # Critic
    #         critic_learning_rate=config[model_type][f"{model_type}_critic_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic learning rate set")
    #         else:
    #             logger.debug(f"critic learning rate set")
    #         critic_optimizer = config[model_type][f"{model_type}_critic_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer set")
    #         else:
    #             logger.debug(f"critic optimizer set")
    #         critic_optimizer_params = {}
    #         if critic_optimizer == "Adam":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            
    #         elif critic_optimizer == "Adagrad":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
            
    #         elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer params set")
    #         else:
    #             logger.debug(f"critic optimizer params set")

    #         critic_normalize_layers = config[model_type][f"{model_type}_critic_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic normalize layers set")
    #         else:
    #             logger.debug(f"critic normalize layers set")
    #         # Set device
    #         device = config[model_type][f"{model_type}_device"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} device set")
    #         else:
    #             logger.debug(f"device set")
    #         # Check if CNN layers and if so, build CNN model
    #         if actor_cnn_layers:
    #             actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
    #         else:
    #             actor_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor cnn layers set: {actor_cnn_layers}")
    #         else:
    #             logger.debug(f"actor cnn layers set: {actor_cnn_layers}")

    #         if critic_cnn_layers:
    #             critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
    #         else:
    #             critic_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic cnn layers set: {critic_cnn_layers}")
    #         else:
    #             logger.debug(f"critic cnn layers set: {critic_cnn_layers}")
    #         # # Get actor clamp value
    #         # clamp_output = config[model_type][f"{model_type}_actor_clamp_output"]
    #         # if comm is not None:
    #         #     logger.debug(f"{comm.Get_name()}; Rank {rank} clamp output set: {clamp_output}")
    #         # else:
    #         #     logger.debug(f"clamp output set: {clamp_output}")
    #         actor_model = models.ActorModel(env = env,
    #                                         cnn_model = actor_cnn_model,
    #                                         dense_layers = actor_layers,
    #                                         output_layer_kernel=kernels[f'actor_output_kernel'],
    #                                         optimizer = actor_optimizer,
    #                                         optimizer_params = actor_optimizer_params,
    #                                         learning_rate = actor_learning_rate,
    #                                         normalize_layers = actor_normalize_layers,
    #                                         # clamp_output=clamp_output,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor model built: {actor_model.get_config()}")
    #         else:
    #             logger.debug(f"actor model built: {actor_model.get_config()}")
    #         critic_model = models.CriticModel(env = env,
    #                                         cnn_model = critic_cnn_model,
    #                                         state_layers = critic_state_layers,
    #                                         merged_layers = critic_merged_layers,
    #                                         output_layer_kernel=kernels[f'critic_output_kernel'],
    #                                         optimizer = critic_optimizer,
    #                                         optimizer_params = critic_optimizer_params,
    #                                         learning_rate = critic_learning_rate,
    #                                         normalize_layers = critic_normalize_layers,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic model built: {critic_model.get_config()}")
    #         else:
    #             logger.debug(f"critic model built: {critic_model.get_config()}")
    #         # get normalizer clip value
    #         normalizer_clip = config[model_type][f"{model_type}_normalizer_clip"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} normalizer clip set: {normalizer_clip}")
    #         else:
    #             logger.debug(f"normalizer clip set: {normalizer_clip}")
    #         # get action epsilon
    #         action_epsilon = config[model_type][f"{model_type}_epsilon_greedy"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} action epsilon set: {action_epsilon}")
    #         else:
    #             logger.debug(f"action epsilon set: {action_epsilon}")
    #         # Replay buffer size
    #         replay_buffer_size = config[model_type][f"{model_type}_replay_buffer_size"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} replay buffer size set: {replay_buffer_size}")
    #         else:
    #             logger.debug(f"replay buffer size set: {replay_buffer_size}")
    #         # Save dir
    #         save_dir = config[model_type][f"{model_type}_save_dir"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} save dir set: {save_dir}")
    #         else:
    #             logger.debug(f"save dir set: {save_dir}")

    #         # create replay buffer
    #         replay_buffer = ReplayBuffer(env, replay_buffer_size, device=device)

    #         # create TD3 agent
    #         td3_agent= cls(
    #             env = env,
    #             actor_model = actor_model,
    #             critic_model = critic_model,
    #             discount = config[model_type][f"{model_type}_discount"],
    #             tau = config[model_type][f"{model_type}_tau"],
    #             action_epsilon = action_epsilon,
    #             replay_buffer = replay_buffer,
    #             batch_size = config[model_type][f"{model_type}_batch_size"],
    #             noise = Noise.create_instance(config[model_type][f"{model_type}_noise"], shape=env.action_space.shape, **config[model_type][f"{model_type}_noise_{config[model_type][f'{model_type}_noise']}"], device=device),
    #             target_noise_stddev = config[model_type][f"{model_type}_target_action_stddev"],
    #             target_noise_clip = config[model_type][f"{model_type}_target_action_clip"],
    #             actor_update_delay = config[model_type][f"{model_type}_actor_update_delay"],
    #             warmup = config[model_type][f"{model_type}_warmup"],
    #             callbacks = callbacks,
    #             comm = comm,
    #             device = device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} TD3 agent built: {td3_agent.get_config()}")
    #         else:
    #             logger.debug(f"TD3 agent built: {td3_agent.get_config()}")
            
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier called")
    #         else:
    #             logger.debug(f"train barrier called")

    #         if comm is not None:
    #             comm.Barrier()
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier passed")

    #         td3_agent.train(
    #                 num_episodes=train_config['num_episodes'],
    #                 render=False,
    #                 render_freq=0,
    #                 )

    #     except Exception as e:
    #         logger.error(f"An error occurred: {e}", exc_info=True)

    # def train(
    #     self, num_episodes, render_freq: int = None, save_dir=None, run_number=None):
    #     """Trains the model for 'episodes' number of episodes."""

    #     # set models to train mode
    #     self.actor_model.train()
    #     self.critic_model_a.train()
    #     self.critic_model_b.train()

    #     # Update save_dir if passed
    #     if save_dir is not None and save_dir.split("/")[-2] != "td3":
    #         self.save_dir = save_dir + "/td3/"
    #         print(f'new save dir: {self.save_dir}')
    #     elif save_dir is not None and save_dir.split("/")[-2] == "td3":
    #         self.save_dir = save_dir
    #         print(f'new save dir: {self.save_dir}')
        
    #     if self.callbacks:
    #         for callback in self.callbacks:
    #                 self._config = callback._config(self)
    #         if self.use_mpi:
    #             if self.rank == 0:
    #                 for callback in self.callbacks:
    #                     if isinstance(callback, rl_callbacks.WandbCallback):
    #                         callback.on_train_begin((self.critic_model_a, self.critic_model_b, self.actor_model,), logs=self._config)
    #                         # logger.debug(f'{self.group}; Rank {self.rank} TD3.train on train begin callback complete')
    #                     else:
    #                         callback.on_train_begin(logs=self._config)
    #         else:
    #             for callback in self.callbacks:
    #                 if isinstance(callback, rl_callbacks.WandbCallback):
    #                     callback.on_train_begin((self.critic_model_a, self.critic_model_b, self.actor_model,), logs=self._config)
    #                     # logger.debug(f'TD3.train on train begin callback complete')
    #                 else:
    #                     callback.on_train_begin(logs=self._config)

        
    #     if self.use_mpi:
    #         try:
    #             # instantiate new environment. Only rank 0 env will render episodes if render==True
    #             if self.rank == 0:
    #                 self.env = self._initialize_env(render, render_freq, context='train')
    #                 # logger.debug(f'{self.group}; Rank {self.rank} initiating environment with render {render}')
    #             else:
    #                 self.env = self._initialize_env(False, 0, context='train')
    #                 # logger.debug(f'{self.group}; Rank {self.rank} initializing environment')
    #         except Exception as e:
    #             logger.error(f"{self.group}; Rank {self.rank} Error in TD3.train agent._initialize_env process: {e}", exc_info=True)
        
    #     else:
    #         try:
    #             # instantiate new environment. Only rank 0 env will render episodes if render==True
    #             self.env = self._initialize_env(render, render_freq, context='train')
    #             # logger.debug(f'initiating environment with render {render}')
    #         except Exception as e:
    #             logger.error(f"Error in TD3.train agent._initialize_env process: {e}", exc_info=True)

    #     # initialize step counter (for logging)
    #     self._step = 1
    #     # set best reward
    #     try:
    #         best_reward = self.env.reward_range[0]
    #     except:
    #         best_reward = -np.inf
    #     # instantiate list to store reward history
    #     reward_history = []
    #     # instantiate lists to store time history
    #     episode_time_history = []
    #     step_time_history = []
    #     learning_time_history = []
    #     steps_per_episode_history = []  # List to store steps per episode

    #     # Calculate total_steps and wait_steps
    #     # max_episode_steps = self.env.spec.max_episode_steps
    #     # total_steps = num_episodes * max_episode_steps
    #     # profiling_steps = (self.profiler_active_steps + self.profiler_warmup_steps) * self.profiler_repeat
    #     # wait_steps = (total_steps - profiling_steps) // self.profiler_repeat

    #     # Profile setup
    #     # with torch.profiler.profile(
    #     #     activities=[
    #     #         torch.profiler.ProfilerActivity.CPU,
    #     #         torch.profiler.ProfilerActivity.CUDA,
    #     #     ],
    #     #     schedule=torch.profiler.schedule(
    #     #         wait=wait_steps,
    #     #         warmup=self.profiler_warmup_steps,
    #     #         active=self.profiler_active_steps,
    #     #         repeat=self.profiler_repeat
    #     #     ),
    #     #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/td3'),
    #     #     record_shapes=True,
    #     #     profile_memory=True,
    #     #     with_stack=True
    #     # ) as prof:
    #     for i in range(num_episodes):
    #         episode_start_time = time.time()
    #         if self.callbacks:
    #             if self.use_mpi:
    #                 if self.rank == 0:
    #                     for callback in self.callbacks:
    #                         callback.on_train_epoch_begin(epoch=self._step, logs=None)
    #                         # logger.debug(f'{self.group}; Rank {self.rank} TD3.train on train epoch begin callback completed')
    #             else:
    #                 for callback in self.callbacks:
    #                     callback.on_train_epoch_begin(epoch=self._step, logs=None)
    #                     # logger.debug(f'TD3.train on train epoch begin callback completed')
    #         # reset noise
    #         if type(self.noise) == OUNoise:
    #             self.noise.reset()
    #         # reset environment
    #         state, _ = self.env.reset()
    #         done = False
    #         episode_reward = 0
    #         episode_steps = 0  # Initialize steps counter for the episode
    #         while not done:
    #             # run callbacks on train batch begin
    #             # if self.callbacks:
    #             #     for callback in self.callbacks:
    #             #         callback.on_train_step_begin(step=self._step, logs=None)
    #             step_start_time = time.time()
    #             action = self.get_action(state)
    #             next_state, reward, term, trunc, _ = self.env.step(action)
    #             # extract observation from next state if next_state is dict (robotics)
    #             if isinstance(next_state, dict):
    #                 next_state = next_state['observation']

    #             # store trajectory in replay buffer
    #             self.replay_buffer.add(state, action, reward, next_state, done)
    #             if term or trunc:
    #                 done = True
    #             episode_reward += reward
    #             state = next_state
    #             episode_steps += 1

    #             # check if enough samples in replay buffer and if so, learn from experiences
    #             if self.replay_buffer.counter > self.batch_size and self.replay_buffer.counter > self.warmup:
    #                 learn_time = time.time()
    #                 actor_loss, critic_loss = self.learn()
    #                 self._train_step_config["actor_loss"] = actor_loss
    #                 self._train_step_config["critic_loss"] = critic_loss

    #                 learning_time_history.append(time.time() - learn_time)

    #             step_time = time.time() - step_start_time
    #             step_time_history.append(step_time)

    #             self._train_step_config["step_reward"] = reward
    #             self._train_step_config["step_time"] = step_time

    #             # log to wandb if using wandb callback
    #             if self.callbacks:
    #                 if self.use_mpi:
    #                     # only have the main process log callback values to avoid multiple callback calls
    #                     if self.rank == 0:
    #                         for callback in self.callbacks:
    #                             callback.on_train_step_end(step=self._step, logs=self._train_step_config)
    #                             # logger.debug(f'{self.group}; Rank {self.rank} TD3.train on train step end callback completed')
    #                 else:
    #                     for callback in self.callbacks:
    #                         callback.on_train_step_end(step=self._step, logs=self._train_step_config)
    #                         # logger.debug(f'TD3.train on train step end callback completed')
                
    #             # prof.step()

    #             if not done:
    #                 self._step += 1

    #         episode_time = time.time() - episode_start_time
    #         episode_time_history.append(episode_time)
    #         reward_history.append(episode_reward)
    #         steps_per_episode_history.append(episode_steps) 
    #         avg_reward = np.mean(reward_history[-100:])
    #         avg_episode_time = np.mean(episode_time_history[-100:])
    #         avg_step_time = np.mean(step_time_history[-100:])
    #         avg_learn_time = np.mean(learning_time_history[-100:])
    #         avg_steps_per_episode = np.mean(steps_per_episode_history[-100:])  # Calculate average steps per episode

    #         self._train_episode_config['episode'] = i
    #         self._train_episode_config["episode_reward"] = episode_reward
    #         self._train_episode_config["avg_reward"] = avg_reward
    #         self._train_episode_config["episode_time"] = episode_time

    #         # check if best reward
    #         if avg_reward > best_reward:
    #             best_reward = avg_reward
    #             self._train_episode_config["best"] = True
    #             # save model
    #             self.save()
    #         else:
    #             self._train_episode_config["best"] = False

    #         if self.callbacks:
    #             if self.use_mpi:
    #                 if self.rank == 0:
    #                     for callback in self.callbacks:
    #                         callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)
    #                         # logger.debug(f'{self.group}; Rank {self.rank} TD3.train on train epoch callback completed')
    #             else:
    #                 for callback in self.callbacks:
    #                     callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)
    #                     # logger.debug(f'TD3.train on train epoch callback completed')

    #         print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}, episode_time {episode_time:.2f}s, avg_episode_time {avg_episode_time:.2f}s, avg_step_time {avg_step_time:.6f}s, avg_learn_time {avg_learn_time:.6f}s, avg_steps_per_episode {avg_steps_per_episode:.2f}")

    #     if self.callbacks:
    #         if self.use_mpi:
    #             if self.rank == 0:
    #                 for callback in self.callbacks:
    #                     callback.on_train_end(logs=self._train_episode_config)
    #                     # logger.debug(f'{self.group}; Rank {self.rank} TD3.train on train end callback complete')
    #         else:
    #             for callback in self.callbacks:
    #                 callback.on_train_end(logs=self._train_episode_config)
    #                 # logger.debug(f'TD3.train on train end callback complete')
    #     # close the environment
    #     self.env.close()

    def train(self, num_episodes: int, num_envs: int, seed: int = None, render_freq: int = 0, sync_interval: int = 1):
        """Trains the TD3 agent for a given number of episodes."""
        # Set models to train mode
        self.actor_model.train()
        self.critic_model_a.train()
        self.critic_model_b.train()
        # Set target models to eval mode
        self.target_actor_model.eval()
        self.target_critic_model_a.eval()
        self.target_critic_model_b.eval()

        self.num_envs = num_envs
        if seed is None:
            seed = np.random.randint(1000)
        set_seed(seed)
        self._sync_interval = sync_interval
        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, WandbCallback):
                    config = self.get_config()
                    config['num_episodes'] = num_episodes
                    config['seed'] = seed
                    config['num_envs'] = self.num_envs
                    config['distributed'] = self._distributed
                    callback.on_train_begin((self.critic_model_a, self.critic_model_b, self.actor_model,), logs=config)
                else:
                    callback.on_train_begin(logs=self._config)
        try:
            # Use the EnvWrapper's _initialize_env method
            self.env.env = self.env._initialize_env(render_freq, num_envs, seed)
        except Exception as e:
            self.logger.error("Error in TD3.train during env initialization", exc_info=True)
        self._step = 0
        best_reward = -np.inf
        score_history = deque(maxlen=100)
        episode_scores = np.zeros(self.num_envs)
        self.completed_episodes = np.zeros(self.num_envs)
        states, _ = self.env.reset()
        while self.completed_episodes.sum() < num_episodes:
            # If distributed, sync to shared agent
            if self._distributed and self._step % self._sync_interval == 0:
                params = self.get_parameters()
                self.apply_parameters(params)
            self._step += 1
            rendered = False # Flag to keep track of render status to avoid rendering multiple times per step
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            actions = self.get_action(states)
            actions = self.env.format_actions(actions)
            next_states, rewards, terms, truncs, _ = self.env.step(actions)
            self._train_step_config["step_reward"] = rewards.mean()
            episode_scores += rewards
            dones = np.logical_or(terms, truncs)
            for i in range(self.num_envs):
                self.replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
                if dones[i]:
                    # increment completed episodes for env by 1
                    self.completed_episodes[i] += 1
                    score_history.append(episode_scores[i])
                    avg_reward = sum(score_history) / len(score_history)
                    self._train_episode_config['episode'] = int(self.completed_episodes.sum())
                    self._train_episode_config['episode_reward'] = episode_scores[i]
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        self._train_episode_config["best"] = 1
                        self.save()
                    else:
                        self._train_episode_config["best"] = 0
                    
                    # Check if number of completed episodes should trigger render
                    if self.completed_episodes.sum() % render_freq == 0 and not rendered:
                        print(f"Rendering episode {self.completed_episodes.sum()} during training...")
                        # Call the test function to render an episode
                        self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
                        # Add render to wandb log
                        video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes.sum()}.mp4")
                        # Log the video to wandb
                        if self.callbacks:
                            for callback in self.callbacks:
                                if isinstance(callback, WandbCallback):
                                    wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")}, step=self._step)
                        rendered = True
                        # Switch models back to train mode after rendering
                        self.actor_model.train()
                        self.critic_model_a.train()
                        self.critic_model_b.train()


                    if self.callbacks:
                        for callback in self.callbacks:
                            callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)
                    print(f"Environment {i}: Episode {int(self.completed_episodes.sum())}, Score {episode_scores[i]}, Avg Score {avg_reward}")
                    episode_scores[i] = 0
            states = next_states
            if self._step > self.warmup and self.replay_buffer.counter > self.batch_size:
                actor_loss, critic_loss = self.learn()
                self._train_step_config["critic_loss"] = critic_loss
                if actor_loss is not None:
                    self._train_step_config["actor_loss"] = actor_loss
                # Step schedulers if not None
                if self.noise_schedule:
                    self.noise_schedule.step()
                    self._train_step_config["noise_anneal"] = self.noise_schedule.get_factor()
                if self.target_noise_schedule:
                    self.target_noise_schedule.step()
                    self._train_step_config["target_noise_anneal"] = self.target_noise_schedule.get_factor()
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_end(step=self._step, logs=self._train_step_config)
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)

       
    # def test(self, num_episodes, render, render_freq):
    #     """Runs a test over 'num_episodes'."""

    #     # set model in eval mode
    #     self.actor_model.eval()
    #     self.critic_model_a.eval()
    #     self.critic_model_b.eval()

    #     # instantiate list to store reward history
    #     reward_history = []
    #     # instantiate new environment
    #     self.env = self._initialize_env(render, render_freq, context='test')
    #     if self.callbacks:
    #         for callback in self.callbacks:
    #             callback.on_test_begin(logs=self._config)

    #     self._step = 1
    #     # set the model to calculate no gradients during evaluation
    #     with T.no_grad():
    #         for i in range(num_episodes):
    #             if self.callbacks:
    #                 for callback in self.callbacks:
    #                     callback.on_test_epoch_begin(epoch=self._step, logs=None) # update to pass any logs if needed
    #             states = []
    #             next_states = []
    #             actions = []
    #             rewards = []
    #             state, _ = self.env.reset()
    #             done = False
    #             episode_reward = 0
    #             while not done:
    #                 action = self.get_action(state, test=True)
    #                 next_state, reward, term, trunc, _ = self.env.step(action)
    #                 # extract observation from next state if next_state is dict (robotics)
    #                 if isinstance(next_state, dict):
    #                     next_state = next_state['observation']
    #                 # store trajectories
    #                 states.append(state)
    #                 actions.append(action)
    #                 next_states.append(next_state)
    #                 rewards.append(reward)
    #                 if term or trunc:
    #                     done = True
    #                 episode_reward += reward
    #                 state = next_state
    #                 self._step += 1
    #             reward_history.append(episode_reward)
    #             avg_reward = np.mean(reward_history[-100:])
    #             self._test_episode_config["episode_reward"] = episode_reward
    #             self._test_episode_config["avg_reward"] = avg_reward
    #             if self.callbacks:
    #                 for callback in self.callbacks:
    #                     callback.on_test_epoch_end(epoch=self._step, logs=self._test_episode_config)

    #             print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

    #         if self.callbacks:
    #             for callback in self.callbacks:
    #                 callback.on_test_end(logs=self._test_episode_config)
    #         # close the environment
    #         self.env.close()
       
    def test(self, num_episodes: int, num_envs: int = 1, seed: int = None, render_freq: int = 0, training: bool = False):
        """Tests the TD3 agent for a given number of episodes."""
        self.actor_model.eval()
        self.critic_model_a.eval()
        self.critic_model_b.eval()
        if seed is None:
            seed = np.random.randint(100)
        if render_freq is None:
            render_freq = 0
        set_seed(seed)
        try:
            env = self.env._initialize_env(render_freq, num_envs, seed)
        except Exception as e:
            self.logger.error("Error in TD3.test during env initialization", exc_info=True)
        if self.callbacks and not training:
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)
        _step = 0
        completed_episodes = np.zeros(num_envs)
        episode_scores = np.zeros(num_envs)
        completed_scores = deque(maxlen=num_episodes)
        frames = []
        states, _ = env.reset()
        while completed_episodes.sum() < num_episodes:
            _step += 1
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_begin(epoch=_step, logs=None)
            actions = self.get_action(states, test=True)
            actions = self.env.format_actions(actions, testing=True)
            next_states, rewards, terms, truncs, _ = env.step(actions)
            self._test_step_config["step_reward"] = rewards
            episode_scores += rewards
            dones = np.logical_or(terms, truncs)
            completed_episodes += dones
            for i in range(num_envs):
                if dones[i]:
                    completed_scores.append(episode_scores[i])
                    self._test_episode_config["episode_reward"] = episode_scores[i]
                    # Save the video if the episode number is divisible by render_freq
                    if (render_freq > 0) and ((completed_episodes.sum()) % render_freq == 0):
                        if training:
                            render_video(frames, self.completed_episodes.sum(), self.save_dir, 'train')
                        else:
                            render_video(frames, completed_episodes.sum(), self.save_dir, 'test')
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/test/episode_{completed_episodes.sum()}.mp4")
                            # Log the video to wandb
                            if self.callbacks:
                                for callback in self.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        wandb.log({"training_video": wandb.Video(video_path, caption="Testing process", format="mp4")})
                        # Empty frames array
                        frames = []
                    if self.callbacks and not training:
                        for callback in self.callbacks:
                            callback.on_test_epoch_end(epoch=_step, logs=self._test_episode_config)
                    print(f"Environment {i}: Episode {int(completed_episodes.sum())}/{num_episodes} Score: {completed_scores[-1]} Avg Score: {sum(completed_scores)/len(completed_scores)}")
                    episode_scores[i] = 0
            if render_freq > 0:
                frame = env.render()[0]
                frames.append(frame)
                states = next_states
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=_step, logs=self._test_step_config)
        if self.callbacks and not training:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)


    def get_config(self):
        return {
            "agent_type": "TD3",
            "env": self.env.to_json(),
            "actor_model": self.actor_model.get_config(),
            "critic_model_a": self.critic_model_a.get_config(),
            "critic_model_b": self.critic_model_b.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "replay_buffer": self.replay_buffer.get_config() if self.replay_buffer is not None else None,
            "batch_size": self.batch_size,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "target_noise": self.target_noise.get_config() if self.target_noise is not None else None,
            "target_noise_schedule": self.target_noise_schedule.get_config() if self.target_noise_schedule is not None else None,
            "target_noise_clip": self.target_noise_clip,
            "actor_update_delay": self.actor_update_delay,
            "grad_clip": self.grad_clip,
            "warmup": self.warmup,
            "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
            "save_dir": self.save_dir,
            "device": self.device.type,
        }

    def save(self):
            
        config = self.get_config()
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.actor_model.save(self.save_dir)
        self.critic_model_a.save(self.save_dir)
        self.critic_model_b.save(self.save_dir)


    # def save(self, save_dir=None):
    #     """Saves the model."""

    #     # Change self.save_dir if save_dir
    #     # if save_dir is not None:
    #     #     self.save_dir = save_dir + "/ddpg/"

    #     config = self.get_config()

    #     # makes directory if it doesn't exist
    #     os.makedirs(self.save_dir, exist_ok=True)

    #     # writes and saves JSON file of DDPG agent config
    #     with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
    #         json.dump(config, f, cls=CustomJSONEncoder)

    #     # saves policy and value model
    #     self.actor_model.save(self.save_dir)
    #     self.critic_model_a.save(self.save_dir)
    #     self.critic_model_b.save(self.save_dir)

    #     if self.normalize_inputs:
    #         self.state_normalizer.save_state(self.save_dir + "state_normalizer.npz")

        # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")


    # @classmethod
    # def load(cls, config, load_weights=True):
    #     """Loads the model."""
    #     # # load reinforce agent config
    #     # with open(
    #     #     Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
    #     # ) as f:
    #     #     config = json.load(f)

    #     # create EnvSpec from config
    #     env_spec_json = json.dumps(config["env"])
    #     env_spec = gym.envs.registration.EnvSpec.from_json(env_spec_json)

    #     # load policy model
    #     actor_model = models.ActorModel.load(config['save_dir'], load_weights)
    #     # load value model
    #     critic_model = models.CriticModel.load(config['save_dir'], load_weights)
    #     # load replay buffer if not None
    #     if config['replay_buffer'] is not None:
    #         config['replay_buffer']['config']['env'] = gym.make(env_spec)
    #         replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
    #     else:
    #         replay_buffer = None
    #     # load noise
    #     noise = Noise.create_instance(config["noise"]["class_name"], **config["noise"]["config"])
    #     # load callbacks
    #     callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]

    #     # return TD3 agent
    #     agent = cls(
    #         gym.make(env_spec),
    #         actor_model = actor_model,
    #         critic_model = critic_model,
    #         discount=config["discount"],
    #         tau=config["tau"],
    #         action_epsilon=config["action_epsilon"],
    #         replay_buffer=replay_buffer,
    #         batch_size=config["batch_size"],
    #         noise=noise,
    #         target_noise_stddev = config['target_noise_stddev'],
    #         target_noise_clip = config['target_noise_clip'],
    #         actor_update_delay = config['actor_update_delay'],
    #         normalize_inputs = config['normalize_inputs'],
    #         normalizer_clip = config['normalizer_clip'],
    #         warmup = config['warmup'],
    #         callbacks=callbacks,
    #         save_dir=config["save_dir"],
    #         device=config["device"],
    #     )

    #     if agent.normalize_inputs:
    #         agent.state_normalizer = Normalizer.load_state(config['save_dir'] + "state_normalizer.npz")

    #     return agent

    @classmethod
    def load(cls, config, load_weights=True):
        """Loads the model."""
        # Load EnvWrapper
        env_wrapper = EnvWrapper.from_json(config["env"])
            
        # load policy model
        actor_model = ActorModel.load(config['actor_model'], load_weights)
        # load value model
        critic_model_a = CriticModel.load(config['critic_model_a'], load_weights)
        critic_model_b = CriticModel.load(config['critic_model_b'], load_weights)
        # load replay buffer if not None
        if config['replay_buffer'] is not None:
            config['replay_buffer']['config']['env'] = env_wrapper
            if config['replay_buffer']['class_name'] == 'PrioritizedReplayBuffer':
                replay_buffer = PrioritizedReplayBuffer(**config["replay_buffer"]["config"])
            else:
                replay_buffer = ReplayBuffer(**config["replay_buffer"]["config"])
        else:
            replay_buffer = None
        noise = Noise.create_instance(config["noise"]["class_name"], **config["noise"]["config"])
        target_noise = Noise.create_instance(config["target_noise"]["class_name"], **config["target_noise"]["config"])
        callbacks = [callback_load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]\
                    if config['callbacks'] else None

        agent = cls(
            env=env_wrapper,
            actor_model=actor_model,
            critic_model_a=critic_model_a,
            critic_model_b=critic_model_b,
            discount=config["discount"],
            tau=config["tau"],
            action_epsilon=config["action_epsilon"],
            replay_buffer=replay_buffer,
            batch_size=config["batch_size"],
            noise=noise,
            noise_schedule=ScheduleWrapper(config["noise_schedule"]),
            target_noise=target_noise,
            target_noise_schedule=ScheduleWrapper(config["target_noise_schedule"]),
            target_noise_clip=config["target_noise_clip"],
            actor_update_delay=config["actor_update_delay"],
            grad_clip=config["grad_clip"],
            warmup=config["warmup"],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
        )
        return agent

class HER(Agent):
    """Hindsight Experience Replay Agent wrapper."""

    def __init__(
        self,
        agent: Agent,
        strategy: str = 'final',
        tolerance: float = 0.5,
        num_goals: int = 4,
        normalizer_clip: float = 5.0,
        normalizer_eps: float = 0.01,
        device: str = None,
        save_dir: str = "models",
        # callbacks: Optional[list[Callback]] = None
    ):
        """
        Initializes the HER agent wrapper.
        
        Args:
            agent (Agent): The underlying agent (e.g., DDPG, TD3) to wrap with HER.
            strategy (str): HER strategy for goal sampling ('final', 'future', etc.).
            tolerance (float): Distance threshold for success determination.
            num_goals (int): Number of goals to sample for hindsight replay.
            normalizer_clip (float): Clipping value for state and goal normalizers.
            normalizer_eps (float): Epsilon for numerical stability in normalizers.
            replay_buffer_size (int): Size of the replay buffer.
            device (str): Device for computation ('cuda' or 'cpu').
            save_dir (str): Directory to save models and logs.
            # callbacks (Optional[list[Callback]]): List of callbacks for training.
        """
        try:
            # Set device
            self.device = T.device("cuda" if device == 'cuda' and T.cuda.is_available() else "cpu")
            self.agent = agent
            self.strategy = strategy
            self.tolerance = tolerance
            self.num_goals = num_goals
            self.normalizer_clip = normalizer_clip
            self.normalizer_eps = normalizer_eps
            # self.replay_buffer_size = replay_buffer_size
            
            # Set save directory
            # if save_dir is not None and "/her/" not in save_dir:
            #     self.save_dir = os.path.join(save_dir, "her")
            #     agent_name = os.path.basename(os.path.dirname(self.agent.save_dir))
            #     self.agent.save_dir = os.path.join(self.save_dir, agent_name)
            # elif save_dir is not None:
            #     self.save_dir = save_dir
            #     agent_name = os.path.basename(os.path.dirname(self.agent.save_dir))
            #     self.agent.save_dir = os.path.join(self.save_dir, agent_name)
            if save_dir is not None and "/her/" not in save_dir:
                self.save_dir = save_dir + "/her/"
            elif save_dir is not None:
                self.save_dir = save_dir

        except Exception as e:
            logger.error(f"Error in HER init: {e}", exc_info=True)

        # Internal attributes
        try:
            obs_space = (self.agent.env.single_observation_space 
                        if hasattr(self.agent.env, "single_observation_space")
                        else self.agent.env.observation_space)
            if isinstance(obs_space, gym.spaces.Dict):
                self._obs_space_shape = obs_space['observation'].shape
                self._goal_shape = obs_space['desired_goal'].shape
            else:
                raise ValueError("HER requires a goal-aware observation space (gym.spaces.Dict)")

            # Initialize HER flag in agent
            self.agent._init_her()
            
            # Set distance threshold based on environment type
            if isinstance(self.agent.env.env, gym.vector.SyncVectorEnv):
                # Vectorized environment: set distance_threshold for each sub-environment
                for i in range(len(self.agent.env.env.envs)):
                    base_env = self.agent.env.get_base_env(i)
                    if hasattr(base_env, "distance_threshold"):
                        base_env.distance_threshold = self.tolerance
                    else:
                        logger.warning(f"Environment {base_env} does not have distance_threshold attribute")
            else:
                # Non-vectorized environment: set directly if attribute exists
                if hasattr(self.agent.env.env, "distance_threshold"):
                    self.agent.env.env.distance_threshold = self.tolerance
                else:
                    logger.warning("Underlying environment does not have distance_threshold attribute")

            # Initialize replay buffer and normalizers
            # self.replay_buffer = ReplayBuffer(
            #     env=self.agent.env,
            #     buffer_size=self.replay_buffer_size,
            #     goal_shape=self._goal_shape,
            #     device=self.device.type,
            # )
            self.state_normalizer = Normalizer(
                self._obs_space_shape, self.normalizer_eps, self.normalizer_clip, self.device
            )
            self.goal_normalizer = Normalizer(
                self._goal_shape, self.normalizer_eps, self.normalizer_clip, self.device
            )

        except Exception as e:
            logger.error(f"Error in HER init internal attributes: {e}", exc_info=True)

        # # Set callbacks
        # try:
        #     self.callbacks = callbacks if callbacks else self.agent.callbacks
        #     if self.callbacks:
        #         for callback in self.callbacks:
        #             self._config = callback._config(self)
        #             if isinstance(callback, WandbCallback):
        #                 self._wandb = True
        #     else:
        #         self._wandb = False
        # except Exception as e:
        #     logger.error(f"Error in HER init callbacks: {e}", exc_info=True)

        # Initialize config dictionaries
        # self._train_config = {}
        # self._train_episode_config = {}
        # self._train_step_config = {}
        # self._test_config = {}
        # self._test_episode_config = {}
        # self._test_step_config = {}
        # self._step = 0


        
    # @classmethod
    # def sweep_train(
    #     cls,
    #     config, # wandb.config,
    #     train_config,
    #     env_spec,
    #     callbacks,
    #     run_number,
    #     comm=None,
    # ):
    #     """Builds and trains agents from sweep configs. Works with MPI"""
    #     rank = MPI.COMM_WORLD.rank

    #     if comm is not None:
    #         logger.debug(f"Rank {rank} comm detected")
    #         rank = comm.Get_rank()
    #         logger.debug(f"Global rank {MPI.COMM_WORLD.Get_rank()} in {comm.Get_name()} set to comm rank {rank}")

    #         logger.debug(f"init_sweep fired: global rank {MPI.COMM_WORLD.rank}, group rank {rank}, {comm.Get_name()}")
    #     else:
    #         logger.debug(f"init_sweep fired: global rank")
    #     try:
    #         # rank = MPI.COMM_WORLD.rank
    #         # Instantiate her variable 
    #         her = None
    #         # Instantiate env from env_spec
    #         env = gym.make(gym.envs.registration.EnvSpec.from_json(env_spec))
    #         # agent_config_path = f'sweep/agent_config_{run_number}.json'
    #         # logger.debug(f"rank {rank} agent config path: {agent_config_path}")
    #         model_type = list(config.keys())[0]
    #         # config = wandb.config
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train config: {train_config}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} env spec id: {env.spec.id}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} callbacks: {callbacks}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} run number: {run_number}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} config set: {config}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} model type: {model_type}")
    #             # Only primary process (rank 0) calls wandb.init() to build agent and log data
    #         else:
    #             logger.debug(f"train config: {train_config}")
    #             logger.debug(f"env spec id: {env.spec.id}")
    #             logger.debug(f"callbacks: {callbacks}")
    #             logger.debug(f"run number: {run_number}")
    #             logger.debug(f"config set: {config}")
    #             logger.debug(f"model type: {model_type}")

    #         actor_cnn_layers, critic_cnn_layers, actor_layers, critic_state_layers, critic_merged_layers, kernels = wandb_support.format_layers(config)
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} layers built")
    #         else:
    #             logger.debug(f"layers built")
    #         # Actor
    #         actor_learning_rate=config[model_type][f"{model_type}_actor_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor learning rate set")
    #         else:
    #             logger.debug(f"actor learning rate set")
    #         actor_optimizer = config[model_type][f"{model_type}_actor_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer set")
    #         else:
    #             logger.debug(f"actor optimizer set")
    #         # get optimizer params
    #         actor_optimizer_params = {}
    #         if actor_optimizer == "Adam":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            
    #         elif actor_optimizer == "Adagrad":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
            
    #         elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
    #             actor_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
    #             actor_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']

    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor optimizer params set")
    #         else:
    #             logger.debug(f"actor optimizer params set")
    #         actor_normalize_layers = config[model_type][f"{model_type}_actor_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor normalize layers set")
    #         else:
    #             logger.debug(f"actor normalize layers set")
    #         # Critic
    #         critic_learning_rate=config[model_type][f"{model_type}_critic_learning_rate"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic learning rate set")
    #         else:
    #             logger.debug(f"critic learning rate set")
    #         critic_optimizer = config[model_type][f"{model_type}_critic_optimizer"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer set")
    #         else:
    #             logger.debug(f"critic optimizer set")
    #         critic_optimizer_params = {}
    #         if critic_optimizer == "Adam":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            
    #         elif critic_optimizer == "Adagrad":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['lr_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
            
    #         elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
    #             critic_optimizer_params['weight_decay'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
    #             critic_optimizer_params['momentum'] = \
    #                 config[model_type][f"{model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic optimizer params set")
    #         else:
    #             logger.debug(f"critic optimizer params set")

    #         critic_normalize_layers = config[model_type][f"{model_type}_critic_normalize_layers"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic normalize layers set")
    #         else:
    #             logger.debug(f"critic normalize layers set")
    #         # Set device
    #         device = config[model_type][f"{model_type}_device"]
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} device set")
    #         else:
    #             logger.debug(f"device set")
    #         # Check if CNN layers and if so, build CNN model
    #         if actor_cnn_layers:
    #             actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
    #         else:
    #             actor_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor cnn layers set: {actor_cnn_layers}")
    #         else:
    #             logger.debug(f"actor cnn layers set: {actor_cnn_layers}")

    #         if critic_cnn_layers:
    #             critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
    #         else:
    #             critic_cnn_model = None
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic cnn layers set: {critic_cnn_layers}")
    #         else:
    #             logger.debug(f"critic cnn layers set: {critic_cnn_layers}")
    #         # get desired, achieved, reward func for env
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} second call env.spec: {env.spec.id}")
    #         else:
    #             logger.debug(f"second call env.spec: {env.spec.id}")
    #         desired_goal_func, achieved_goal_func, reward_func = gym_helper.get_her_goal_functions(env)
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} goal function set")
    #         else:
    #             logger.debug(f"goal function set")
    #         # Reset env state to initiate state to detect correct goal shape
    #         _,_ = env.reset()
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} env reset")
    #         else:
    #             logger.debug(f"env reset")
    #         goal_shape = desired_goal_func(env).shape
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} goal shape set: {goal_shape}")
    #         else:
    #             logger.debug(f"goal shape set: {goal_shape}")
    #         # Get actor clamp value
    #         # clamp_output = config[model_type][f"{model_type}_actor_clamp_output"]
    #         # logger.debug(f"{comm.Get_name()}; Rank {rank} clamp output set: {clamp_output}")
    #         actor_model = models.ActorModel(env = env,
    #                                         cnn_model = actor_cnn_model,
    #                                         dense_layers = actor_layers,
    #                                         output_layer_kernel=kernels[f'actor_output_kernel'],
    #                                         goal_shape=goal_shape,
    #                                         optimizer = actor_optimizer,
    #                                         optimizer_params = actor_optimizer_params,
    #                                         learning_rate = actor_learning_rate,
    #                                         normalize_layers = actor_normalize_layers,
    #                                         # clamp_output=clamp_output,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} actor model built: {actor_model.get_config()}")
    #         else:
    #             logger.debug(f"actor model built: {actor_model.get_config()}")
    #         critic_model = models.CriticModel(env = env,
    #                                         cnn_model = critic_cnn_model,
    #                                         state_layers = critic_state_layers,
    #                                         merged_layers = critic_merged_layers,
    #                                         output_layer_kernel=kernels[f'critic_output_kernel'],
    #                                         goal_shape=goal_shape,
    #                                         optimizer = critic_optimizer,
    #                                         optimizer_params = critic_optimizer_params,
    #                                         learning_rate = critic_learning_rate,
    #                                         normalize_layers = critic_normalize_layers,
    #                                         device=device,
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} critic model built: {critic_model.get_config()}")
    #         else:
    #             logger.debug(f"critic model built: {critic_model.get_config()}")
    #         # get goal metrics
    #         strategy = config[model_type][f"{model_type}_goal_strategy"]
            
    #         tolerance = config[model_type][f"{model_type}_goal_tolerance"]
            
    #         num_goals = config[model_type][f"{model_type}_num_goals"]
            
    #         # get normalizer clip value
    #         normalizer_clip = config[model_type][f"{model_type}_normalizer_clip"]
            
    #         # get action epsilon
    #         action_epsilon = config[model_type][f"{model_type}_epsilon_greedy"]
            
    #         # Replay buffer size
    #         replay_buffer_size = config[model_type][f"{model_type}_replay_buffer_size"]
            
    #         # Save dir
    #         save_dir = config[model_type][f"{model_type}_save_dir"]
            
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} strategy set: {strategy}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} tolerance set: {tolerance}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} num goals set: {num_goals}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} normalizer clip set: {normalizer_clip}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} action epsilon set: {action_epsilon}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} replay buffer size set: {replay_buffer_size}")
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} save dir set: {save_dir}")
    #         else:
    #             logger.debug(f"strategy set: {strategy}")
    #             logger.debug(f"tolerance set: {tolerance}")
    #             logger.debug(f"num goals set: {num_goals}")
    #             logger.debug(f"normalizer clip set: {normalizer_clip}")
    #             logger.debug(f"action epsilon set: {action_epsilon}")
    #             logger.debug(f"replay buffer size set: {replay_buffer_size}")
    #             logger.debug(f"save dir set: {save_dir}")
            
            
    #         if model_type == "HER_DDPG":
    #             ddpg_agent= DDPG(
    #                 env = env,
    #                 actor_model = actor_model,
    #                 critic_model = critic_model,
    #                 discount = config[model_type][f"{model_type}_discount"],
    #                 tau = config[model_type][f"{model_type}_tau"],
    #                 action_epsilon = action_epsilon,
    #                 replay_buffer = None,
    #                 batch_size = config[model_type][f"{model_type}_batch_size"],
    #                 noise = Noise.create_instance(config[model_type][f"{model_type}_noise"], shape=env.action_space.shape, **config[model_type][f"{model_type}_noise_{config[model_type][f'{model_type}_noise']}"], device=device),
    #                 callbacks = callbacks,
    #                 comm = comm
    #             )
    #             if comm is not None:
    #                 logger.debug(f"{comm.Get_name()}; Rank {rank} ddpg agent built: {ddpg_agent.get_config()}")
    #             else:
    #                 logger.debug(f"ddpg agent built: {ddpg_agent.get_config()}")

    #         elif model_type == "HER_TD3":
    #             ddpg_agent= TD3(
    #                 env = env,
    #                 actor_model = actor_model,
    #                 critic_model = critic_model,
    #                 discount = config[model_type][f"{model_type}_discount"],
    #                 tau = config[model_type][f"{model_type}_tau"],
    #                 action_epsilon = action_epsilon,
    #                 replay_buffer = None,
    #                 batch_size = config[model_type][f"{model_type}_batch_size"],
    #                 noise = Noise.create_instance(config[model_type][f"{model_type}_noise"], shape=env.action_space.shape, **config[model_type][f"{model_type}_noise_{config[model_type][f'{model_type}_noise']}"], device=device),
    #                 target_noise_stddev= config[model_type][f"{model_type}_target_action_stddev"],
    #                 target_noise_clip= config[model_type][f"{model_type}_target_action_clip"],
    #                 actor_update_delay= config[model_type][f"{model_type}_actor_update_delay"],
    #                 callbacks = callbacks,
    #                 comm = comm
    #             )
    #             if comm is not None:
    #                 logger.debug(f"{comm.Get_name()}; Rank {rank} ddpg agent built: {ddpg_agent.get_config()}")
    #             else:
    #                 logger.debug(f"ddpg agent built: {ddpg_agent.get_config()}")

    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} build barrier called")
    #             comm.Barrier()
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} build barrier passed")

    #         her = cls(
    #             agent = ddpg_agent,
    #             strategy = strategy,
    #             tolerance = tolerance,
    #             num_goals = num_goals,
    #             desired_goal = desired_goal_func,
    #             achieved_goal = achieved_goal_func,
    #             reward_fn = reward_func,
    #             normalizer_clip = normalizer_clip,
    #             replay_buffer_size = replay_buffer_size,
    #             device = device,
    #             save_dir = save_dir,
    #             comm = comm
    #         )
    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} her agent built: {her.get_config()}")
    #         else:
    #             logger.debug(f"her agent built: {her.get_config()}")

    #         if comm is not None:
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier called")
    #             comm.Barrier()
    #             logger.debug(f"{comm.Get_name()}; Rank {rank} train barrier passed")

    #         her.train(
    #                 num_epochs=train_config['num_epochs'],
    #                 num_cycles=train_config['num_cycles'],
    #                 num_episodes=train_config['num_episodes'],
    #                 num_updates=train_config['num_updates'],
    #                 render=False,
    #                 render_freq=0,
    #                 )

    #     except Exception as e:
    #         logger.error(f"An error occurred: {e}", exc_info=True)

    # def train(self, num_episodes: int, num_updates: int, render_freq: int, num_envs: int = 1, seed: int = None):
    #     """
    #     Train the HER agent with a vectorized environment setup.

    #     Args:
    #         num_epochs (int): Number of training epochs.
    #         num_cycles (int): Number of cycles per epoch.
    #         num_episodes (int): Number of episodes per cycle.
    #         num_updates (int): Number of learning updates per step when buffer is sufficiently large.
    #         render (bool): Whether to render the environment.
    #         render_freq (int): Frequency of rendering.
    #         save_dir (str, optional): Directory to save models.
    #         num_envs (int): Number of parallel environments (default: 1).
    #         seed (int, optional): Random seed for reproducibility.
    #     """
    #     try:
    #         logger.debug("HER train fired")

    #         # Set models to train mode
    #         self.agent.actor_model.train()
    #         if hasattr(self.agent, 'critic_model'):
    #             self.agent.critic_model.train()  # For DDPG
    #         if hasattr(self.agent, 'critic_model_a'):
    #             self.agent.critic_model_a.train()  # For TD3
    #         if hasattr(self.agent, 'critic_model_b'):
    #             self.agent.critic_model_b.train()  # For TD3

    #         # Update agent config
    #         if self.agent.callbacks:
    #             self.agent._config.update({
    #                 # 'num_epochs': num_epochs,
    #                 # 'num_cycles': num_cycles,
    #                 'num_episodes': num_episodes,
    #                 'num_updates': num_updates,
    #                 'tolerance': self.tolerance,
    #                 'num_envs': num_envs,
    #                 'seed': seed
    #             })
    #             logger.debug("HER.train: train config added to agent config")

    #         # Initialize callbacks
    #         if self.agent.callbacks:
    #             for callback in self.agent.callbacks:
    #                 if isinstance(callback, WandbCallback):
    #                     models = (self.agent.critic_model, self.agent.actor_model)
    #                     if isinstance(self.agent, TD3):
    #                         models = (self.agent.critic_model_a, self.agent.critic_model_b, self.agent.actor_model)
    #                     callback.on_train_begin(models, logs=self.agent._config)
    #                 else:
    #                     callback.on_train_begin(logs=self.agent._config)

    #         # Initialize environment
    #         try:
    #             self.agent.env.env = self.agent.env._initialize_env(render_freq, num_envs, seed)
    #             logger.debug(f"Initializing environment with render_freq={render_freq}, num_envs={num_envs}, seed={seed}")
    #         except Exception as e:
    #             logger.error(f"Error in HER.train environment initialization: {e}", exc_info=True)

    #         # Initialize counters and histories
    #         self.agent._step = 0  # Use agent's step counter for consistency with TD3/DDPG
    #         self.completed_episodes = np.zeros(num_envs)
    #         episode_scores = np.zeros(num_envs)
    #         score_history = deque(maxlen=100)
    #         trajectory_buffers = [[] for _ in range(num_envs)]
    #         best_reward = -np.inf
    #         success_counter = 0.0

    #         states, _ = self.agent.env.reset()

    #         # Total episodes across all environments
    #         # total_episodes = num_epochs * num_cycles * num_episodes
    #         while self.completed_episodes.sum() < num_episodes:
    #             self.agent._step += 1
    #             rendered = False # Flag to keep track of render status to avoid rendering multiple times per step

    #             if self.agent.callbacks:
    #                 for callback in self.agent.callbacks:
    #                     callback.on_train_epoch_begin(epoch=self.agent._step, logs=None)

    #             # Get actions for all environments
    #             actions = self.agent.get_action(
    #                 states['observation'],
    #                 states['desired_goal'],
    #                 test=False,  # Training mode
    #                 state_normalizer=self.state_normalizer,
    #                 goal_normalizer=self.goal_normalizer
    #             )
    #             # Format actions for vectorized environment
    #             formatted_actions = self.agent.env.format_actions(actions)
    #             # Step the environment
    #             next_states, rewards, terms, truncs, _ = self.agent.env.step(formatted_actions)
    #             #DEBUG
    #             # if np.any(rewards == 0.0):
    #             #     print(f"0 reward found at step {self.agent._step}: {rewards}")
    #             dones = np.logical_or(terms, truncs)
    #             episode_scores += rewards

    #             for i in range(num_envs):
    #                 # Add original transition to replay buffer
    #                 self.agent.replay_buffer.add(
    #                     states['observation'][i],
    #                     actions[i],
    #                     rewards[i],
    #                     next_states['observation'][i],
    #                     dones[i],
    #                     states['achieved_goal'][i],
    #                     next_states['achieved_goal'][i],
    #                     states['desired_goal'][i]
    #                 )

    #                 # Append transition to trajectory buffer
    #                 trajectory_buffers[i].append({
    #                     'state': states['observation'][i],
    #                     'action': actions[i],
    #                     'reward': rewards[i],
    #                     'next_state': next_states['observation'][i],
    #                     'done': dones[i],
    #                     'achieved_goal': states['achieved_goal'][i],
    #                     'next_achieved_goal': next_states['achieved_goal'][i],
    #                     'desired_goal': states['desired_goal'][i]
    #                 })

    #                 # Update normalizer local stats with obs and goal info
    #                 self.state_normalizer.update_local_stats(
    #                     T.tensor(states['observation'][i], dtype=T.float32,
    #                              device=self.state_normalizer.device.type)
    #                 )

    #                 self.goal_normalizer.update_local_stats(
    #                     T.tensor(states['achieved_goal'][i], dtype=T.float32,
    #                              device=self.state_normalizer.device.type)
    #                 )

    #                 # calculate success rate
    #                 goal_distance = np.linalg.norm(states['achieved_goal'][i] - states['desired_goal'][i], axis=-1)
    #                 # goal_distance = self.agent.env.get_base_env().goal_distance(states['achieved_goal'][i], states['desired_goal'][i])
    #                 success = (goal_distance <= self.tolerance).astype(np.float32)
    #                 # success = self.agent.env.get_base_env(i)._is_success(states['achieved_goal'][i], states['desired_goal'][i])
    #                 #DEBUG
    #                 # print(f'goal distance:{goal_distance}')
    #                 # print(f'success:{success}')
    #                 success_counter += success
    #                 # print(f'success counter:{success_counter}')
    #                 # To correctly calculate success percentage, must divide success counter by
    #                 # num envs to put on same scale as self.agent._step
    #                 success_perc = (success_counter / num_envs) / self.agent._step
    #                 # store success metrics to train step config
    #                 self.agent._train_step_config["success rate"] = success_perc
    #                 self.agent._train_step_config["goal distance"] = goal_distance

    #                 if dones[i]:
    #                     # Apply hindsight to the completed trajectory
    #                     self.store_hindsight_trajectory(trajectory_buffers[i])
    #                     # Reset trajectory buffer
    #                     trajectory_buffers[i] = []
    #                     # Increment completed episodes
    #                     self.completed_episodes[i] += 1
    #                     # Update score history
    #                     score_history.append(episode_scores[i])
    #                     avg_reward = np.mean(score_history) if score_history else 0
    #                     self.agent._train_episode_config.update({
    #                         'episode': int(self.completed_episodes.sum()),
    #                         'episode_reward': episode_scores[i],
    #                         # 'avg_reward': avg_reward
    #                     })
    #                     # Check for best reward and save
    #                     if avg_reward > best_reward:
    #                         best_reward = avg_reward
    #                         self.agent._train_episode_config["best"] = 1
    #                         self.save()
    #                     else:
    #                         self.agent._train_episode_config["best"] = 0

    #                     if self.agent.callbacks:
    #                         for callback in self.agent.callbacks:
    #                             callback.on_train_epoch_end(epoch=self.agent._step, logs=self.agent._train_episode_config)

    #                     # Check if number of completed episodes should trigger render
    #                     if self.completed_episodes.sum() % render_freq == 0 and not rendered:
    #                         print(f"Rendering episode {self.completed_episodes.sum()} during training...")
    #                         # Call the test function to render an episode
    #                         self.test(num_episodes=1, seed=None, render_freq=1, training=True)
    #                         # Add render to wandb log
    #                         video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes.sum()}.mp4")
    #                         # Log the video to wandb
    #                         if self.agent.callbacks:
    #                             for callback in self.agent.callbacks:
    #                                 if isinstance(callback, WandbCallback):
    #                                     wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")}, step=self.agent._step)
    #                         rendered = True
    #                         # Set models to train mode
    #                         self.agent.actor_model.train()
    #                         if hasattr(self.agent, 'critic_model'):
    #                             self.agent.critic_model.train()  # For DDPG
    #                         if hasattr(self.agent, 'critic_model_a'):
    #                             self.agent.critic_model_a.train()  # For TD3
    #                         if hasattr(self.agent, 'critic_model_b'):
    #                             self.agent.critic_model_b.train()  # For TD3

    #                     print(f"Environment {i}: episode {int(self.completed_episodes[i])}, score {episode_scores[i]}, avg_score {avg_reward}")
    #                     episode_scores[i] = 0

    #             # Update normalizer global values
    #             self.state_normalizer.update_global_stats()
    #             self.goal_normalizer.update_global_stats()

    #             # Perform learning updates
    #             if self.agent._step > self.agent.warmup:
    #                 if self.agent.replay_buffer.counter > self.agent.batch_size:
    #                     for _ in range(num_updates):
    #                         actor_loss, critic_loss = self.agent.learn(
    #                             # replay_buffer=self.replay_buffer,
    #                             state_normalizer=self.state_normalizer,
    #                             goal_normalizer=self.goal_normalizer
    #                         )
    #                         self.agent._train_step_config.update({
    #                             "actor_loss": actor_loss,
    #                             "critic_loss": critic_loss
    #                         })
    #                         # if isinstance(self.agent, DDPG):
    #                         #     self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
    #                         #     self.agent.soft_update(self.agent.critic_model, self.agent.target_critic_model)

    #                     # Step scheduler if not None
    #                     if self.agent.noise_schedule:
    #                         self.agent.noise_schedule.step()
    #                         self.agent._train_step_config["noise_anneal"] = self.agent.noise_schedule.get_factor()

    #             # Update states
    #             states = next_states

    #             # Log step metrics
    #             self.agent._train_step_config["step_reward"] = rewards.mean()
    #             if self.agent.callbacks:
    #                 for callback in self.agent.callbacks:
    #                     callback.on_train_step_end(step=self.agent._step, logs=self.agent._train_step_config)

    #         if self.agent.callbacks:
    #             for callback in self.agent.callbacks:
    #                 callback.on_train_end(logs=self.agent._train_episode_config)

    #         self.agent.env.close()

    #     except Exception as e:
    #         logger.error(f"Error during HER train process: {e}", exc_info=True)

    def train(self, num_epochs: int, num_cycles: int, num_episodes_per_cycle: int, num_updates: int, render_freq: int, num_envs: int = 1, seed: int = None):
        """
        Train the HER agent with a vectorized environment setup, following the HER paper's experiment structure.

        Args:
            num_epochs (int): Number of training epochs.
            num_cycles (int): Number of cycles per epoch.
            num_episodes_per_cycle (int): Number of episodes to collect per cycle across all environments.
            num_updates (int): Number of optimization steps per cycle after collecting episodes.
            render_freq (int): Frequency of rendering (in total completed episodes).
            num_envs (int): Number of parallel environments (default: 1).
            seed (int, optional): Random seed for reproducibility.
        """
        try:
            logger.debug("HER train fired")

            # Set models to train mode
            self.agent.actor_model.train()
            if hasattr(self.agent, 'critic_model'):
                self.agent.critic_model.train()  # For DDPG
            if hasattr(self.agent, 'critic_model_a'):
                self.agent.critic_model_a.train()  # For TD3
            if hasattr(self.agent, 'critic_model_b'):
                self.agent.critic_model_b.train()  # For TD3

            # Update agent config
            if self.agent.callbacks:
                self.agent._config.update({
                    'strategy': self.strategy,
                    'num_goals': self.num_goals if self.strategy == 'future' else None,
                    'num_epochs': num_epochs,
                    'num_cycles': num_cycles,
                    'num_episodes_per_cycle': num_episodes_per_cycle,
                    'num_updates': num_updates,
                    'tolerance': self.tolerance,
                    'num_envs': num_envs,
                    'seed': seed
                })
                logger.debug("HER.train: train config added to agent config")

            # Initialize callbacks
            if self.agent.callbacks:
                for callback in self.agent.callbacks:
                    if isinstance(callback, WandbCallback):
                        if isinstance(self.agent, DDPG):
                            models = (self.agent.critic_model, self.agent.actor_model)
                        elif isinstance(self.agent, TD3):
                            models = (self.agent.critic_model_a, self.agent.critic_model_b, self.agent.actor_model)
                        callback.on_train_begin(models, logs=self.agent._config)
                    else:
                        callback.on_train_begin(logs=self.agent._config)

            # Initialize environment
            try:
                self.agent.env.env = self.agent.env._initialize_env(render_freq, num_envs, seed)
                logger.debug(f"Initializing environment with render_freq={render_freq}, num_envs={num_envs}, seed={seed}")
            except Exception as e:
                logger.error(f"Error in HER.train environment initialization: {e}", exc_info=True)

            # Initialize counters and histories
            self.agent._step = 0
            self.completed_episodes = np.zeros(num_envs)
            episode_scores = np.zeros(num_envs)
            score_history = deque(maxlen=100)
            trajectories = [[] for _ in range(num_envs)]
            best_reward = -np.inf
            success_counter = 0.0

            states, _ = self.agent.env.reset()

            # Training loop
            for epoch in range(num_epochs):
                if self.agent.callbacks:
                    for callback in self.agent.callbacks:
                        callback.on_train_epoch_begin(epoch=epoch, logs=None)

                for cycle in range(num_cycles):
                    completed_before_cycle = self.completed_episodes.sum()
                    rendered = False  # Reset render flag per cycle

                    # Collect episodes until num_episodes_per_cycle are completed
                    while self.completed_episodes.sum() < completed_before_cycle + num_episodes_per_cycle:
                        self.agent._step += 1

                        # Get actions for all environments
                        actions = self.agent.get_action(
                            states['observation'],
                            states['desired_goal'],
                            test=False,
                            state_normalizer=self.state_normalizer,
                            goal_normalizer=self.goal_normalizer
                        )
                        formatted_actions = self.agent.env.format_actions(actions)
                        next_states, rewards, terms, truncs, _ = self.agent.env.step(formatted_actions)
                        dones = np.logical_or(terms, truncs)
                        episode_scores += rewards
                        # Store transitions in the env trajectory
                        for i in range(num_envs):
                            trajectories[i].append(
                                (
                                    states['observation'][i],
                                    actions[i],
                                    rewards[i],
                                    next_states['observation'][i],
                                    dones[i],
                                    states['achieved_goal'][i],
                                    next_states['achieved_goal'][i],
                                    states['desired_goal'][i]
                                )
                            )

                            # Update normalizers
                            self.state_normalizer.update_local_stats(
                                T.tensor(states['observation'][i], dtype=T.float32, device=self.state_normalizer.device.type)
                            )
                            self.goal_normalizer.update_local_stats(
                                T.tensor(states['achieved_goal'][i], dtype=T.float32, device=self.goal_normalizer.device.type)
                            )

                        completed_episodes = np.flatnonzero(dones) # Get indices of completed episodes
                        for i in completed_episodes:
                        # for i in range(num_envs):
                            #DEBUG
                            # print(f'trajectories[{i}]: {trajectories[i]}')
                            self.store_hindsight_trajectory(trajectories[i])
                            # Calculate success rate
                            goal_distance = np.linalg.norm(states['achieved_goal'][i] - states['desired_goal'][i], axis=-1)
                            success = (goal_distance <= self.tolerance).astype(np.float32)
                            success_counter += success
                            success_perc = (success_counter / self.completed_episodes.sum())
                            self.agent._train_step_config.update({
                                "success rate": success_perc,
                                "goal distance": goal_distance,
                                "step_reward": rewards.mean()
                            })
                            trajectories[i] = []
                            self.completed_episodes[i] += 1
                            # Add original transition to replay buffer
                            # self.agent.replay_buffer.add(
                            #     states['observation'][i],
                            #     actions[i],
                            #     rewards[i],
                            #     next_states['observation'][i],
                            #     dones[i],
                            #     states['achieved_goal'][i],
                            #     next_states['achieved_goal'][i],
                            #     states['desired_goal'][i]
                            # )

                            # Append transition to trajectory buffer
                            score_history.append(episode_scores[i])
                            avg_reward = np.mean(score_history) if score_history else 0
                            self.agent._train_episode_config.update({
                                'episode': int(self.completed_episodes.sum()),
                                'episode_reward': episode_scores[i],
                            })

                            if avg_reward > best_reward:
                                best_reward = avg_reward
                                self.agent._train_episode_config["best"] = 1
                                self.save()
                            else:
                                self.agent._train_episode_config["best"] = 0

                            if self.agent.callbacks:
                                for callback in self.agent.callbacks:
                                    callback.on_train_epoch_end(epoch=self.agent._step, logs=self.agent._train_episode_config)

                            if self.completed_episodes.sum() % render_freq == 0 and not rendered:
                                print(f"Rendering episode {self.completed_episodes.sum()} during training...")
                                self.test(num_episodes=1, seed=None, render_freq=1, training=True)
                                video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.completed_episodes.sum()}.mp4")
                                if self.agent.callbacks:
                                    for callback in self.agent.callbacks:
                                        if isinstance(callback, WandbCallback):
                                            wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")}, step=self.agent._step)
                                rendered = True
                                # Reset models to train mode
                                self.agent.actor_model.train()
                                if hasattr(self.agent, 'critic_model'):
                                    self.agent.critic_model.train()
                                if hasattr(self.agent, 'critic_model_a'):
                                    self.agent.critic_model_a.train()
                                if hasattr(self.agent, 'critic_model_b'):
                                    self.agent.critic_model_b.train()

                            print(f"Environment {i}: episode {int(self.completed_episodes[i])}, score {episode_scores[i]}, avg_score {avg_reward}")
                            episode_scores[i] = 0


                        

                            # if dones[i]:
                            #     # Apply hindsight to the completed trajectory
                            #     self.store_hindsight_trajectory(trajectory_buffers[i])
                            #     trajectory_buffers[i] = []
                            #     self.completed_episodes[i] += 1

                        states = next_states

                        if self.agent.callbacks:
                            for callback in self.agent.callbacks:
                                callback.on_train_step_end(step=self.agent._step, logs=self.agent._train_step_config)
                                
                    # Update normalizers and states
                    self.state_normalizer.update_global_stats()
                    self.goal_normalizer.update_global_stats()

                    # Perform optimization after collecting episodes
                    if self.agent._step > self.agent.warmup:
                        if self.agent.replay_buffer.counter > self.agent.batch_size:
                            for _ in range(num_updates):
                                actor_loss, critic_loss = self.agent.learn(
                                    state_normalizer=self.state_normalizer,
                                    goal_normalizer=self.goal_normalizer
                                )
                                self.agent._train_step_config.update({
                                    "actor_loss": actor_loss,
                                    "critic_loss": critic_loss
                                })
                            # Update target networks
                            if isinstance(self.agent, DDPG):
                                self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
                                self.agent.soft_update(self.agent.critic_model, self.agent.target_critic_model)
                            elif isinstance(self.agent, TD3):
                                self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
                                self.agent.soft_update(self.agent.critic_model_a, self.agent.target_critic_model_a)
                                self.agent.soft_update(self.agent.critic_model_b, self.agent.target_critic_model_b)

                            # Step noise scheduler if not None
                            if self.agent.noise_schedule:
                                self.agent.noise_schedule.step()
                                self.agent._train_step_config["noise_anneal"] = self.agent.noise_schedule.get_factor()
                            # Step target noise scheduler if is attr and not None
                            if hasattr(self.agent, 'target_noise_schedule') and self.agent.target_noise_schedule:
                                self.agent.target_noise_schedule.step()
                                self.agent._train_step_config["target_noise_anneal"] = self.agent.target_noise_schedule.get_factor()

                if self.agent.callbacks:
                    for callback in self.agent.callbacks:
                        callback.on_train_epoch_end(epoch=epoch, logs=self.agent._train_episode_config)

            if self.agent.callbacks:
                for callback in self.agent.callbacks:
                    callback.on_train_end(logs=self.agent._train_episode_config)

            self.agent.env.close()

        except Exception as e:
            logger.error(f"Error during HER train process: {e}", exc_info=True)
    
    def test(self, num_episodes: int, num_envs: int = 1, seed: int = None, render_freq: int = 0, training: bool = False):
        """Runs a test over 'num_episodes'."""

        # Set models to eval mode
        self.agent.actor_model.eval()
        if hasattr(self.agent, 'critic_model'):
            self.agent.critic_model.eval()  # For DDPG
        if hasattr(self.agent, 'critic_model_a'):
            self.agent.critic_model_a.eval()  # For TD3
        if hasattr(self.agent, 'critic_model_b'):
            self.agent.critic_model_b.eval()  # For TD3
        
        if seed is None:
            seed = np.random.randint(10000)
        if render_freq is None:
            render_freq = 0
        set_seed(seed)

        try:
            env = self.agent.env._initialize_env(render_freq, num_envs, seed)
        except Exception as e:
            logger.error("Error in HER.test during env initialization", exc_info=True)

        if self.agent.callbacks and not training:
            for callback in self.agent.callbacks:
                self.agent._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    self.agent._config['seed'] = seed
                    self.agent._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)
        _step = 0
        completed_episodes = np.zeros(num_envs)
        episode_scores = np.zeros(num_envs)
        completed_scores = deque(maxlen=num_episodes)
        frames = []
        states, _ = env.reset()
        while completed_episodes.sum() < num_episodes:
            _step += 1
            if self.agent.callbacks and not training:
                for callback in self.agent.callbacks:
                    callback.on_test_epoch_begin(epoch=_step, logs=None)
            # Get actions
            actions = self.agent.get_action(
                    states['observation'],
                    states['desired_goal'],
                    test=True,  # Test mode
                    state_normalizer=self.state_normalizer,
                    goal_normalizer=self.goal_normalizer
                )
            actions = self.agent.env.format_actions(actions, testing=True)
            next_states, rewards, terms, truncs, _ = env.step(actions)
            self.agent._test_step_config["step_reward"] = rewards
            episode_scores += rewards
            dones = np.logical_or(terms, truncs)
            completed_episodes += dones
            for i in range(num_envs):
                if dones[i]:
                    completed_scores.append(episode_scores[i])
                    self.agent._test_episode_config["episode_reward"] = episode_scores[i]
                    # Save the video if the episode number is divisible by render_freq
                    if (render_freq > 0) and ((completed_episodes.sum()) % render_freq == 0):
                        if training:
                            render_video(frames, self.completed_episodes.sum(), self.save_dir, 'train')
                        else:
                            render_video(frames, completed_episodes.sum(), self.save_dir, 'test')
                            # Add render to wandb log
                            video_path = os.path.join(self.save_dir, f"renders/test/episode_{completed_episodes.sum()}.mp4")
                            # Log the video to wandb
                            if self.agent.callbacks:
                                for callback in self.agent.callbacks:
                                    if isinstance(callback, WandbCallback):
                                        wandb.log({"training_video": wandb.Video(video_path, caption="Testing process", format="mp4")})
                        # Empty frames array
                        frames = []
                    if self.agent.callbacks and not training:
                        for callback in self.agent.callbacks:
                            callback.agent.on_test_epoch_end(epoch=_step, logs=self.agent._test_episode_config)
                    
                    print(f"Environment {i}: Episode {int(completed_episodes.sum())}/{num_episodes} Score: {completed_scores[-1]} Avg Score: {sum(completed_scores)/len(completed_scores)}")
                    episode_scores[i] = 0

            if render_freq > 0:
                frame = env.render()[0]
                frames.append(frame)
            states = next_states
            if self.agent.callbacks and not training:
                for callback in self.agent.callbacks:
                    callback.on_test_step_end(step=_step, logs=self.agent._test_step_config)
        if self.agent.callbacks and not training:
            for callback in self.agent.callbacks:
                callback.on_test_end(logs=self.agent._test_episode_config)

    def store_hindsight_trajectory(self, trajectory):
        """
        Store hindsight-augmented transitions from a completed trajectory into the replay buffer.
        
        Args:
            trajectory (list): List of dictionaries, each containing transition data with keys:
                'state', 'action', 'reward', 'next_state', 'done', 'achieved_goal',
                'next_achieved_goal', 'desired_goal'
        """
        states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals = zip(*trajectory)

        # # Extract values from the list of dictionaries into separate lists
        # states = [t['state'] for t in trajectory]
        # actions = [t['action'] for t in trajectory]
        # rewards = [t['reward'] for t in trajectory]
        # next_states = [t['next_state'] for t in trajectory]
        # dones = [t['done'] for t in trajectory]
        # achieved_goals = [t['achieved_goal'] for t in trajectory]
        # next_achieved_goals = [t['next_achieved_goal'] for t in trajectory]
        # desired_goals = [t['desired_goal'] for t in trajectory]

        # # Convert lists to NumPy arrays for efficiency
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        achieved_goals = np.array(achieved_goals)
        next_achieved_goals = np.array(next_achieved_goals)
        desired_goals = np.array(desired_goals)

        

        #DEBUG
        # print(f'states shape: {states.shape}')
        # print(f'unique states: {len(np.unique(states))}')
        # print(f'states: {states}')
        # print(f'actions shape: {actions.shape}')
        # print(f'unique actions: {len(np.unique(actions))}')
        # print(f'rewards shape: {rewards.shape}')
        # print(f'unique rewards: {len(np.unique(rewards))}')
        # print(f'next_states shape: {next_states.shape}')
        # print(f'unique next_states: {len(np.unique(next_states))}')
        # print(f'dones shape: {dones.shape}')
        # print(f'unique dones: {len(np.unique(dones))}')
        # print(f'achieved_goals shape: {achieved_goals.shape}')
        # print(f'unique achieved_goals: {len(np.unique(achieved_goals))}')
        # print(f'next_achieved_goals shape: {next_achieved_goals.shape}')
        # print(f'unique next_achieved_goals: {len(np.unique(next_achieved_goals))}')
        # print(f'desired_goals shape: {desired_goals.shape}')
        # print(f'unique desired_goals: {len(np.unique(desired_goals))}')

        # Add actual experiences to the replay buffer
        self.agent.replay_buffer.add(*zip(*trajectory))

        tol_count = 0
        experiences = [] # Store experiences for hindsight

        # loop over each step in the trajectory to set new achieved goals, calculate new reward, and save to replay buffer
        for idx, (state, action, next_state, done, state_achieved_goal, next_state_achieved_goal, desired_goal) in enumerate(zip(states, actions, next_states, dones, achieved_goals, next_achieved_goals, desired_goals)):

            if self.strategy == "final":
                new_desired_goal = next_achieved_goals[-1]
                # new_reward, within_tol = self.reward_fn(self.agent.env, action, state_achieved_goal, next_state_achieved_goal, new_desired_goal, self.tolerance)
                new_reward = self.agent.env.get_base_env().compute_reward(state_achieved_goal, new_desired_goal, {})
                within_tol = self.agent.env.get_base_env()._is_success(state_achieved_goal, new_desired_goal)
                # increment tol_count
                tol_count += within_tol

                # store non normalized trajectory
                experiences.append((state, action, new_reward, next_state, done, state_achieved_goal, next_state_achieved_goal, new_desired_goal))

            elif self.strategy == 'future':
                for i in range(self.num_goals):
                    if idx + i >= len(states) -1:
                        break
                    goal_idx = np.random.randint(idx + 1, len(states))
                    new_desired_goal = next_achieved_goals[goal_idx]
                    # new_reward, within_tol = self.reward_fn(self.agent.env, action, state_achieved_goal, next_state_achieved_goal, new_desired_goal, self.tolerance)
                    new_reward = self.agent.env.get_base_env().compute_reward(state_achieved_goal, new_desired_goal, {})
                    within_tol = self.agent.env.get_base_env()._is_success(state_achieved_goal, new_desired_goal)
                    tol_count += within_tol
                    # store non normalized trajectory
                    experiences.append((state, action, new_reward, next_state, done, state_achieved_goal, next_state_achieved_goal, new_desired_goal))

            elif self.strategy == 'none':
                break

        #DEBUG
        # trajectory = zip(*experiences)
        # print(f'trajectory: {list(trajectory)}')

        self.agent.replay_buffer.add(*zip(*experiences))

        # add tol count to train step config for callbacks
        if self.agent.callbacks:
            self.agent._train_episode_config["tolerance count"] = tol_count
                
        

    def set_normalizer_state(self, config):
        self.agent.state_normalizer.set_state(config)

    # def cleanup(self):
    #     self.replay_buffer.cleanup()
    #     self.state_normalizer.cleanup()
    #     self.goal_normalizer.cleanup()
    #     T.cuda.empty_cache()
    #     if dist.is_initialized():
    #         dist.destroy_process_group()
    #         print("Process group destroyed")
    #     print("Cleanup complete")


    def get_config(self):
        config = {
            "agent_type": self.__class__.__name__,
            "agent": self.agent.get_config(),
            "strategy": self.strategy,
            "tolerance":self.tolerance,
            "num_goals": self.num_goals,
            "normalizer_clip": self.normalizer_clip,
            "normalizer_eps": self.normalizer_eps,
            # "replay_buffer_size": self.replay_buffer_size,
            "device": self.device.type,
            "save_dir": self.save_dir,
        }

        return config
    
    def save(self):
        """Saves the model."""

        # Change self.save_dir if save_dir 
        # if save_dir is not None:
        #     self.save_dir = save_dir + "/her/"
        #     print(f'new save dir: {self.save_dir}')

        config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of HER agent config
        with open(self.save_dir + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        # save agent
        # if save_dir is not None:
        #     self.agent.save(self.save_dir)
        #     print(f'new agent save dir: {self.agent.save_dir}')
        # else:
        self.agent.save()

        self.state_normalizer.save_state(self.save_dir + "state_normalizer.npz")
        self.goal_normalizer.save_state(self.save_dir + "goal_normalizer.npz")

    @classmethod
    def load(cls, config, load_weights=True):
        """Loads the model."""
        # logger.debug(f'rank {MPI.COMM_WORLD.rank} HER.load called')
        # Resolve function names to actual functions
        # try:
        #     config["desired_goal"] = getattr(gym_helper, config["desired_goal"])
        #     config["achieved_goal"] = getattr(gym_helper, config["achieved_goal"])
        #     config["reward_fn"] = getattr(gym_helper, config["reward_fn"])
        #     logger.debug(f"rank {MPI.COMM_WORLD.rank} HER.load successfully loaded gym goal functions")
        # except Exception as e:
        #     logger.error(f"rank {MPI.COMM_WORLD.rank} HER.load failed to load gym goal functions: {e}", exc_info=True)

        # load agent
        try:
            agent = load_agent_from_config(config["agent"], load_weights)
        except Exception as e:
            logger.error(f"HER.load failed to load Agent: {e}", exc_info=True)

        # instantiate HER model
        try:
            her = cls(agent, config["strategy"], config["tolerance"], config["num_goals"],
                    config['normalizer_clip'], config['normalizer_eps'],
                    config["device"], config["save_dir"])
            logger.debug(f"HER.load successfully loaded HER")
        except Exception as e:
            logger.error(f"HER.load failed to load HER: {e}", exc_info=True)

        # load agent normalizers
        try:
            agent.state_normalizer = Normalizer.load_state(config['save_dir'] + "state_normalizer.npz")
            agent.goal_normalizer = Normalizer.load_state(config['save_dir'] + "goal_normalizer.npz")
            logger.debug(f"HER.load successfully loaded normalizers")
        except Exception as e:
            logger.error(f"HER.load failed to load normalizers: {e}", exc_info=True)
        
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
        normalize_values (bool): Whether to normalize value outputs.
        value_norm_clip (float): Clipping range for value normalization.
        policy_grad_clip (float): Maximum norm for policy gradients.
        value_grad_clip (float): Maximum norm for value model gradients
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
                 normalize_values: bool = False,
                 value_normalizer_clip: float = float('inf'),
                 policy_grad_clip: float = float('inf'),
                 value_grad_clip: float = float('inf'),
                 reward_clip: float = float('inf'),
                 callbacks: Optional[list[Callback]] = None,
                 save_dir: str = 'models',
                 device: str = None
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
            normalize_values (bool): Whether to normalize value outputs (default: False).
            value_normalizer_clip (float): Clipping range for value normalization (default: inf).
            policy_grad_clip (float): Maximum norm for policy gradients (default: inf).
            reward_clip (float): Maximum absolute value for reward clipping (default: inf).
            callbacks (list): List of callback objects for logging and monitoring (default: []).
            save_dir (str): Directory to save models and configurations (default: 'models').
            device (str): Device for computations ('cpu' or 'cuda', default: 'cuda').
        """
        self.env = env
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
        self.normalize_values = normalize_values
        self.value_norm_clip = value_normalizer_clip
        self.policy_grad_clip = policy_grad_clip
        self.value_grad_clip = value_grad_clip
        self.reward_clip = reward_clip
        # self.callbacks = callbacks
        self.device = get_device(device)

        if self.normalize_values:
            self.normalizer = Normalizer((1), clip_range=self.value_norm_clip, device=device)

        if save_dir is not None and "/ppo/" not in save_dir:
            self.save_dir = save_dir + "/ppo/"
        elif save_dir is not None:
            self.save_dir = save_dir

        # Initialize callback configurations
        self._initialize_callbacks(callbacks)
        self._train_config = {}
        self._train_episode_config = {}
        self._train_step_config = {}
        self._test_config = {}
        self._test_step_config = {}
        self._test_episode_config = {}
        self._step = None

    def _initialize_callbacks(self, callbacks):
        """
        Initialize and configure callbacks for logging and monitoring.

        Args:
            callbacks (list): List of callback objects.
        """
        try:
            self.callbacks = callbacks
            if callbacks:
                for callback in self.callbacks:
                    self._config = callback._config(self)
                    if isinstance(callback, WandbCallback):
                        self._wandb = True
            else:
                self.callback_list = None
                self._wandb = False
        except Exception as e:
            logger.error(f"Error initializing callbacks: {e}", exc_info=True)

    def calculate_advantages_and_returns(self, rewards, states, next_states, dones):
        """
        Compute advantages and returns using GAE, correctly handling episode terminations.
        """
        #DEBUG
        # print(f'states shape:{states.shape}')
        # print(f'next states shape:{next_states.shape}')
        # print(f'rewards shape:{rewards.shape}')
        # print(f'dones shape:{dones.shape}')
        num_steps, num_envs = rewards.shape
        all_advantages = []
        all_returns = []
        all_values = []

        for env_idx in range(num_envs):
            with T.no_grad():
                rewards_env = rewards[:, env_idx]
                states_env = states[:, env_idx, ...]
                next_states_env = next_states[:, env_idx, ...]
                dones_env = dones[:, env_idx]
                values = self.value_model(states_env).squeeze(-1)
                next_values = self.value_model(next_states_env).squeeze(-1)
                advantages = T.zeros_like(rewards_env)
                returns = T.zeros_like(rewards_env)
                gae = 0.0

                #DEBUG
                # print(f'reward_env shape:{rewards_env.shape}')
                
                # Calculate deltas across the trajectory
                deltas = rewards_env + self.discount * next_values * (1.0 - dones_env) - values
                #DEBUG
                # print(f'deltas shape:{deltas.shape}')

                for t in reversed(range(num_steps)):
                    # delta = rewards_env[t] + self.discount * next_values[t] * (1-) 
                    gae = deltas[t] + self.discount * self.gae_coefficient * gae * (1.0 - dones_env[t])
                    #DEBUG
                    # print(f'gae:{gae}')
                    advantages[t] = gae
                    returns[t] = gae + values[t]

                all_advantages.append(advantages)
                all_returns.append(returns)
                all_values.append(values)

        # Stack results across environments
        advantages = T.stack(all_advantages, dim=1)
        returns = T.stack(all_returns, dim=1)
        values = T.stack(all_values, dim=1)

        self._train_episode_config["values"] = values.mean().item()
        self._train_episode_config["advantages"] = advantages.mean().item()
        self._train_episode_config["returns"] = returns.mean().item()

        # Normalize advantages if required
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        return advantages, returns, values

    def get_action(self, states):
        """
        Select an action based on the current policy.

        Args:
            states (array): Input states.
        
        Returns:
            Tuple[array, array]: Selected actions and their log probabilities.
        """
        with T.no_grad():
            states = T.tensor(states, dtype=T.float32, device=self.policy_model.device)
            #DEBUG
            # print(f'get action states:{states.shape}')
            if self.policy_model.distribution == 'categorical':
                dist, logits = self.policy_model(states)
            else:
                dist, _, _ = self.policy_model(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            actions = actions.detach().cpu().numpy()
            log_probs = log_probs.detach().cpu().numpy()
        return actions, log_probs

    def action_adapter(self, actions):
        """
        Adapt actions to match the environment's action space.

        Args:
            actions (array): Actions to adapt.

        Returns:
            array: Adapted actions.
        """
        if isinstance(self.env, GymnasiumWrapper):
            if isinstance(self.env.single_action_space, gym.spaces.Box):
                action_space_low = self.env.single_action_space.low
                action_space_high = self.env.single_action_space.high
                # Map action values to be between 0-1 if using normal distribution
                if self.policy_model.distribution == 'normal':
                    actions = 1/(1 + np.exp(-actions))
                # Map from [0, 1] to [action_space_low, action_space_high]
                adapted_actions = action_space_low + (action_space_high - action_space_low) * actions
                return adapted_actions
            elif isinstance(self.env.single_action_space, gym.spaces.Discrete):
                n = self.env.single_action_space.n
                # Map actions from [0, 1] to [0, n-1]
                adapted_actions = (actions * n).astype(int)
                adapted_actions = np.clip(adapted_actions, 0, n - 1)
                return adapted_actions
        elif isinstance(self.env, IsaacSimWrapper):
            pass
        else:
            raise NotImplementedError(f"Action adaptation not implemented for environment type: {type(self.env)}")

        raise NotImplementedError("Unsupported action space type for the current environment")
    

    def clip_reward(self, reward):
        """
        Clip rewards to the specified range.

        Args:
            reward (float): Reward to clip.

        Returns:
            float: Clipped reward.
        """
        if reward > self.reward_clip:
            return self.reward_clip
        elif reward < -self.reward_clip:
            return -self.reward_clip
        else:
            return reward

    def train(self, timesteps, trajectory_length, batch_size, learning_epochs, num_envs, seed=None, render_freq:int=0):
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
        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        if render_freq == None:
            render_freq = 0

        # Set seeds
        set_seed(seed)
        # gym.utils.seeding.np_random.seed = seed # Seeds of envs now set in _initialize_env

        if self.callbacks:
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    self._config['timesteps'] = timesteps
                    self._config['trajectory_length'] = trajectory_length
                    self._config['batch_size'] = batch_size
                    self._config['learning_epochs'] = learning_epochs
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                    callback.on_train_begin((self.value_model, self.policy_model,), logs=self._config)
                    # logger.debug(f'TD3.train on train begin callback complete')
                else:
                    callback.on_train_begin(logs=self._config)

        try:
            # instantiate new vec environment
            self.env.env = self.env._initialize_env(0, num_envs, seed)
        except Exception as e:
            logger.error(f"Error in PPO.train agent._initialize_env process: {e}", exc_info=True)

        # set best reward
        best_reward = -np.inf

        self.trajectory_length = trajectory_length
        self.num_envs = num_envs
        self.policy_model.train()
        self.value_model.train()
        self._step = 0
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        # policy_loss_history = []
        # value_loss_history = []
        # entropy_history = []
        # kl_history = []
        # time_history = []
        # param_history = []
        frames = []  # List to store frames for the video
        self.episodes = np.zeros(self.num_envs) # Tracks current episode for each env
        episode_lengths = np.zeros(self.num_envs) # Tracks step count for each env
        scores = np.zeros(self.num_envs) # Tracks current score for each env
        env_scores = np.zeros(self.num_envs)  # Tracks last episode score for each env
        episode_scores = deque(maxlen=100) # Tracks the last 100 episode scores
        states, _ = self.env.reset()

        # set an episode rendered flag to track if an episode has yet to be rendered
        episode_rendered = False
        # track the previous episode number of the first env for rendering
        prev_episode = self.episodes[0]

        while self._step < timesteps:
            self._step += 1
            episode_lengths += 1 # increment the step count of each episode of each env by 1
            dones = []
            actions, log_probs = self.get_action(states)
            #DEBUG
            # print(f'train actions shape:{actions.shape}')
            # print(f'train actions:{actions}')
            if self.policy_model.distribution == 'beta':
                acts = self.action_adapter(actions)
            else:
                acts = actions
            # if self.policy_model.distribution != 'categorical':
            #     acts = acts.astype(np.float32)
            #     acts = acts.tolist()
            #     acts = [[float(a) for a in act] for act in acts]
            acts = self.env.format_actions(acts)
            #DEBUG
            # print(f'acts shape:{acts.shape}')
            # print(f'acts:{acts}')

            # If using WANDB log action values of first environment
            if self.callbacks:
                for callback in self.callbacks:
                    if isinstance(callback, WandbCallback):
                        if self.policy_model.distribution != 'categorical':
                            for i, a in enumerate(acts[0]):
                                self._train_step_config[f'action_{i}'] = a
                        else:
                            self._train_step_config['action'] = acts

            next_states, rewards, terms, truncs, _ = self.env.step(acts)
            # Update scores of each episode
            scores += rewards

            self._train_step_config["step_reward"] = rewards.max()

            for i, (term, trunc) in enumerate(zip(terms, truncs)):
                if term or trunc:
                    dones.append(True)
                    env_scores[i] = scores[i]  # Store score at end of episode
                    episode_scores.append(scores[i]) # Store score in deque to compute avg
                    self._train_step_config["episode_reward"] = scores[i]
                    scores[i] = 0  # Reset score for this environment
                    self._train_step_config["episode_length"] = episode_lengths[i]
                    episode_lengths[i]  = 0 # Resets the step count of the env that returned term/trunc to 0
                else:
                    dones.append(False)

            self.episodes += dones
            # set episode rendered to false if episode number has changed
            if prev_episode != self.episodes[0]:
                episode_rendered = False
            self._train_episode_config['episode'] = self.episodes[0]
            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            clipped_rewards = [self.clip_reward(reward) for reward in rewards]
            all_rewards.append(clipped_rewards)
            all_next_states.append(next_states)
            all_dones.append(dones)

            # render episode if first env shows done and first env episode num % render_freq == 0
            if render_freq > 0 and self.episodes[0] % render_freq == 0 and episode_rendered == False:
                print(f"Rendering episode {self.episodes[0]} during training...")
                # Call the test function to render an episode
                _ = self.test(num_episodes=1, seed=seed, render_freq=1, training=True)
                # Add render to wandb log
                video_path = os.path.join(self.save_dir, f"renders/train/episode_{self.episodes[0]}.mp4")
                # Log the video to wandb
            if self.callbacks:
                for callback in self.callbacks:
                        if isinstance(callback, WandbCallback):
                            wandb.log({"training_video": wandb.Video(video_path, caption="Training process", format="mp4")})
                episode_rendered = True
                # Switch models back to train mode after rendering
                self.policy_model.train()
                self.value_model.train()

            prev_episode = self.episodes[0]

            # env_scores = np.array([
            #     env_score[-1] if len(env_score) > 0 else np.nan
            #     for env_score in episode_scores
            # ])

            if self._step % self.trajectory_length == 0:
                # print(f'learning timestep: {self._step}')
                trajectory = (all_states, all_actions, all_log_probs, all_rewards, all_next_states, all_dones)
                # Get policy clip
                policy_clip = self.policy_clip
                if self.policy_clip_schedule:
                    policy_clip *= self.policy_clip_schedule.get_factor()                    
                self._train_episode_config["policy_clip"] = policy_clip
                # Get value clip
                value_clip = self.value_clip
                if self.value_clip_schedule:
                    value_clip *= self.value_clip_schedule.get_factor()                    
                self._train_episode_config["value_clip"] = value_clip
                # Get entropy coefficient
                entropy_coefficient = self.entropy_coefficient
                if self.entropy_schedule:
                    entropy_coefficient *= self.entropy_schedule.get_factor() 
                self._train_episode_config["entropy_coefficient"] = entropy_coefficient
                # get kl coefficient
                kl_coefficient = self.kl_coefficient
                if self.kl_adapter:
                    kl_coefficient *= self.kl_adapter.get_beta()
                self._train_episode_config["kl_coefficient"] = kl_coefficient
                
                if self.policy_model.distribution == 'categorical':
                    policy_loss, value_loss, entropy, kl, logits = self.learn(trajectory, batch_size, learning_epochs)
                else:
                    policy_loss, value_loss, entropy, kl, param1, param2 = self.learn(trajectory, batch_size, learning_epochs)
                # self._train_episode_config[f"avg_env_scores"] = np.nanmean(env_scores)
                self._train_episode_config["actor_loss"] = policy_loss
                self._train_episode_config["critic_loss"] = value_loss
                self._train_episode_config["entropy"] = entropy
                self._train_episode_config["kl_divergence"] = kl
                if self.policy_model.scheduler:
                    self._train_episode_config['policy learning rate'] = self.policy_model.scheduler.get_last_lr()[0]
                else:
                    self._train_episode_config['policy learning rate'] = self.policy_model.optimizer.param_groups[0]['lr']
                if self.value_model.scheduler:
                    self._train_episode_config['value learning rate'] = self.value_model.scheduler.get_last_lr()[0]
                else:
                    self._train_episode_config['value learning rate'] = self.value_model.optimizer.param_groups[0]['lr']
                # if self.policy_model.distribution == 'categorical':
                #     # Convert logits to probabilities
                #     probabilities = F.softmax(logits, dim=0)
                #     self._train_episode_config["probabilities"] = probabilities
                # else:
                #     self._train_episode_config["param1"] = param1.mean()
                #     self._train_episode_config["param2"] = param2.mean()

                # policy_loss_history.append(policy_loss)
                # value_loss_history.append(value_loss)
                # entropy_history.append(entropy)
                # kl_history.append(kl)
                # if self.policy_model.distribution == 'categorical':
                #     param_history.append(logits)
                # else:
                #     param_history.append((param1, param2))
                
                # Clear trajectory data
                all_states = []
                all_actions = []
                all_log_probs = []
                all_rewards = []
                all_next_states = []
                all_dones = []
                
        if self.callbacks:
            for callback in self.callbacks:
                        callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

                # # Clear CUDA cache
                # T.cuda.empty_cache()

            # Set avg score if 1 or more episodes scores are logged, else set avg to -inf
            if len(episode_scores) > 0:
                avg_score = sum(episode_scores) / len(episode_scores) # compute avg scores
            else:
                avg_score = -np.inf
            # check if best reward
            if avg_score > best_reward:
                best_reward = avg_score
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            states = next_states

            if self._step % 1000 == 0:
                print(f'episode: {self.episodes}; total steps: {self._step}; episodes scores: {env_scores}; avg score: {avg_score}')

        if self.callbacks:
            for callback in self.callbacks:
                    callback.on_train_step_end(step=self._step, logs=self._train_step_config)

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)

        # return {
        #         'scores': episode_scores,
        #         'policy loss': policy_loss_history,
        #         'value loss': value_loss_history,
        #         'entropy': entropy_history,
        #         'kl': kl_history,
        #         'params': param_history,
        #         }

    def learn(self, trajectory, batch_size, learning_epochs):
        """
        Perform learning updates using the collected trajectory.

        Args:
            trajectory (Tuple): Collected trajectory containing states, actions, etc.
            batch_size (int): Batch size for training.
            learning_epochs (int): Number of epochs per update.

        Returns:
            Tuple: policy loss, value loss, entropy, and KL divergence.
        """
        # Unpack trajectory
        all_states, all_actions, all_log_probs, all_rewards, all_next_states, all_dones = trajectory

        # Convert lists to tensors without flattening
        # This results in tensors of shape (num_steps, num_envs, ...)
        states = T.stack([T.tensor(s, dtype=T.float32, device=self.policy_model.device) for s in all_states])
        #DEBUG
        # print(f'learn states shape:{states.shape}')
        # Convert actions to T.long values if categorical, else floats
        if self.policy_model.distribution == 'categorical':
            actions = T.stack([T.tensor(a, dtype=T.long, device=self.policy_model.device) for a in all_actions])
            #DEBUG
            # print(f'actions:{actions}')
        else:
            actions = T.stack([T.tensor(a, dtype=T.float32, device=self.policy_model.device) for a in all_actions])
        log_probs = T.stack([T.tensor(lp, dtype=T.float32, device=self.policy_model.device) for lp in all_log_probs])
        rewards = T.stack([T.tensor(r, dtype=T.float32, device=self.value_model.device) for r in all_rewards])
        next_states = T.stack([T.tensor(ns, dtype=T.float32, device=self.policy_model.device) for ns in all_next_states])
        dones = T.stack([T.tensor(d, dtype=T.int, device=self.policy_model.device) for d in all_dones])

        # Calculate advantages and returns
        advantages, returns, all_values = self.calculate_advantages_and_returns(rewards, states, next_states, dones)

        # Flatten the tensors along the time and environment dimensions for batching
        num_steps, num_envs = rewards.shape
        total_samples = num_steps * num_envs

        # Reshape observations
        # obs_shape = states.shape[2:]  # Get observation shape
        states = states.reshape(total_samples, *self.env.single_observation_space.shape)
        next_states = next_states.reshape(total_samples, *self.env.single_observation_space.shape)
        #DEBUG
        # print(f'learn reshaped states shape:{states.shape}')

        # Reshape tensors for batching
        all_values = all_values.reshape(total_samples, -1) # Shape: (total_samples, 1)
        actions = actions.reshape(total_samples, -1)     # Shape: (total_samples, action_space)
        log_probs = log_probs.reshape(total_samples, -1) # Shape: (total_samples, action_dim)
        advantages = advantages.reshape(total_samples, 1) # Shape: (total_samples, 1)
        returns = returns.reshape(total_samples, 1)      # Shape: (total_samples, 1)
        #DEBUG
        # print(f'resized values:{all_values.shape}')
        # print(f'resized actions:{actions.shape}')
        # print(f'resized log probs:{log_probs.shape}')
        # print(f'resized advantages:{advantages.shape}')
        # print(f'resized returns:{returns.shape}')

        # Set previous distribution to none (used for KL divergence calculation)
        # prev_dist = None

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
            scheduler_params = self.policy_model.scheduler_params,
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
            scheduler_params = self.value_model.scheduler_params,
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
                actions_batch = actions[batch_indices]
                log_probs_batch = log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                

                # Create new distribution
                if self.policy_model.distribution == 'categorical':
                    #DEBUG
                    # New distribution
                    new_dist, logits = self.policy_model(states_batch)
                    new_log_probs = new_dist.log_prob(actions_batch.view(-1))
                    # Old distribution
                    old_dist, old_logits = old_policy(states_batch)
                    old_log_probs = old_dist.log_prob(actions_batch.view(-1))
                    #DEBUG
                    # print(f'new logits: {logits}')
                else: # Continuous Distributions
                    # New distribution
                    new_dist, param1, param2 = self.policy_model(states_batch)
                    new_log_probs = new_dist.log_prob(actions_batch).sum(dim=-1)
                    # Old distribution
                    old_dist, old_param1, old_param2 = old_policy(states_batch)
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
                entropy = new_dist.entropy().mean()
                entropy_penalty = entropy * -entropy_coefficient 

                # Calculate the KL penalty
                kl = kl_divergence(old_dist, new_dist).mean()
                kl_penalty = kl * kl_coefficient
                
                policy_loss = surrogate_loss + entropy_penalty + kl_penalty
                
                # Update the policy
                self.policy_model.optimizer.zero_grad()
                policy_loss.backward()
                T.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.policy_grad_clip)
                self.policy_model.optimizer.step()
                
                    
                # Update the value function
                values = self.value_model(states_batch)
                loss = (values - returns_batch).pow(2)
                old_values = old_value_model(states_batch)
                clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
                clipped_value_loss = (clipped_values - returns_batch).pow(2)
                value_loss = self.value_loss_coefficient * (0.5 * T.max(loss, clipped_value_loss).mean()).mean()
                self.value_model.optimizer.zero_grad()
                value_loss.backward()
                T.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=self.value_grad_clip)
                self.value_model.optimizer.step()

                
        # Step schedulers
        if self.policy_model.scheduler:
            self.policy_model.scheduler.step()
        if self.value_model.scheduler:
            self.value_model.scheduler.step()
        if self.policy_clip_schedule:
            self.policy_clip_schedule.step()
            policy_clip = self.policy_clip * self.policy_clip_schedule.get_factor()
        if self.value_clip_schedule:
            self.value_clip_schedule.step()
            value_clip = self.value_clip * self.value_clip_schedule.get_factor()
        if self.entropy_schedule:
            self.entropy_schedule.step()
            entropy_coefficient = self.entropy_coefficient * self.entropy_schedule.get_factor()
        if self.kl_adapter:
            self.kl_adapter.step(kl)
            kl_coefficient = self.kl_coefficient * self.kl_adapter.get_beta()

        # Create 3d scatter plot of visited states colored by state value and action magnitude
        # if self.callbacks:
        #     for callback in self.callbacks:
        #         if isinstance(callback, WandbCallback):
        #             # Reduce states to 3D embeddings
        #             reducer = UMAP(n_components=3, random_state=42)
        #             embeddings = reducer.fit_transform(states.cpu().numpy())  # Shape: (num_samples, 3)
        #             # Compute the magnitude of the actions
        #             action_magnitude = np.linalg.norm(actions.cpu().numpy(), axis=1)
        #             df = pd.DataFrame({
        #                 'embedding_x': embeddings[:, 0],
        #                 'embedding_y': embeddings[:, 1],
        #                 'embedding_z': embeddings[:, 2],
        #                 'value': all_values.cpu().numpy().flatten(),
        #                 'action_magnitude': action_magnitude,
        #                 # If you want to include specific action components:
        #                 # 'action_component_0': actions[:, 0],
        #                 # 'action_component_1': actions[:, 1],
        #                 # ...
        #             })

        #             # Create a 3D scatter plot colored by value estimates
        #             fig_value = px.scatter_3d(
        #                 df,
        #                 x='embedding_x',
        #                 y='embedding_y',
        #                 z='embedding_z',
        #                 color='value',
        #                 title='State Embeddings Colored by Value Function',
        #                 labels={'embedding_x': 'Embedding X', 'embedding_y': 'Embedding Y', 'embedding_z': 'Embedding Z', 'value': 'Value Estimate'},
        #                 opacity=0.7
        #             )
                    
        #             # Create a 3D scatter plot colored by action magnitude
        #             fig_action = px.scatter_3d(
        #                 df,
        #                 x='embedding_x',
        #                 y='embedding_y',
        #                 z='embedding_z',
        #                 color='action_magnitude',
        #                 title='State Embeddings Colored by Action Magnitude',
        #                 labels={'embedding_x': 'Embedding X', 'embedding_y': 'Embedding Y', 'embedding_z': 'Embedding Z', 'action_magnitude': 'Action Magnitude'},
        #                 opacity=0.7
        #             )

        #             # Log the 3D plots
        #             wandb.log({
        #                 "Value Function Embeddings 3D": fig_value,
        #                 "Policy Embeddings 3D": fig_action
        #             })

        # Decay Policy Clip
        # self.policy_clip *= self.clip_decay
        # Decay Entropy Coefficient
        # self.entropy_coefficient *= self.entropy_decay

        # print(f'Policy Loss: {policy_loss.sum()}')
        # print(f'Value Loss: {value_loss}')
        # print(f'Entropy: {entropy}')
        # print(f'KL Divergence: {kl}')

        if self.policy_model.distribution == 'categorical':
            return policy_loss, value_loss, entropy, kl, logits.detach().cpu().flatten()
        else:
            return policy_loss, value_loss, entropy, kl, param1.detach().cpu().flatten(), param2.detach().cpu().flatten()

    def test(self, num_episodes, num_envs:int=1, seed=None, render_freq:int=0, training: bool=False):
        """
        Test the PPO agent in the environment.

        Args:
            num_episodes (int): Number of episodes to test.
            num_envs (int): Number of parallel environments.
            seed (int, optional): Random seed for reproducibility.
            render_freq (int): Frequency of rendering episodes.
            training (bool): Whether testing is during training.

        Returns:
            dict: Test metrics including scores and log probabilities.
        """
        # Set the policy and value function models to evaluation mode
        self.policy_model.eval()
        self.value_model.eval()

        if seed is None:
            seed = np.random.randint(100)

        # Set render freq to 0 if None is passed
        if render_freq == None:
            render_freq = 0

        # Set seeds
        set_seed(seed)

        env = self.env._initialize_env(render_freq, num_envs, seed)
        if self.callbacks and not training:
            print('test begin callback if statement fired')
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, WandbCallback):
                    # Add to config to send to wandb for logging
                    self._config['seed'] = seed
                    self._config['num_envs'] = num_envs
                callback.on_test_begin(logs=self._config)

        # episode_scores = [[] for _ in range(num_envs)]  # Track scores for each env
        # reset step counter
        step = 0
        all_scores = []
        all_log_probs = []

        for episode in range(num_episodes):
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_begin(epoch=step, logs=None)
            done = False
            states, _ = env.reset()
            scores = 0
            log_probs = []
            frames = []  # List to store frames for the video

            while not done:

                # Get action and log probability from the current policy
                actions, log_prob = self.get_action(states)
                # print(f'actions:{actions}')
                if self.policy_model.distribution == 'beta':
                    acts = self.action_adapter(actions)
                    # print(f'formatted actions from beta:{acts}')
                else:
                    acts = actions
                # if self.policy_model.distribution != 'categorical':
                #     acts = acts.astype(np.float32)
                #     acts = np.clip(acts, env.single_action_space.low, env.single_action_space.high)
                #     acts = acts.tolist()
                #     acts = [[float(a) for a in act] for act in acts]
                acts = self.env.format_actions(acts, testing=True)

                #  log prob to log probs list
                log_probs.append(log_prob)

                # Step the environment
                next_states, rewards, terms, truncs, _ = env.step(acts)
                # Update scores of each episode
                scores += rewards

                for i, (term, trunc) in enumerate(zip(terms, truncs)):
                    if term or trunc:
                        done = True
                        # print(f'append true')
                    # else:
                    #     dones.append(False)

                if render_freq > 0:
                    # Capture the frame
                    frame = env.render()[0]
                    # print(f'frame:{frame}')
                    frames.append(frame)

                # Move to the next state
                states = next_states

                # Add metrics to test step config to log
                self._test_step_config['step_reward'] = rewards[0]
                if self.callbacks and not training:
                    for callback in self.callbacks:
                        callback.on_test_step_end(step=step, logs=self._test_step_config)

                # Increment step count
                step += 1

            # Save the video if the episode number is divisible by render_freq
            if (render_freq > 0) and ((episode + 1) % render_freq == 0):
                if training:
                    render_video(frames, self.episodes[0], self.save_dir, 'train')
                else:
                    render_video(frames, episode+1, self.save_dir, 'test')
                    # Add render to wandb log
                    video_path = os.path.join(self.save_dir, f"renders/test/episode_{episode + 1}.mp4")
                    # Log the video to wandb
                if self.callbacks:
                    for callback in self.callbacks:
                            if isinstance(callback, WandbCallback):
                                wandb.log({"training_video": wandb.Video(video_path, caption="Testing process", format="mp4")})

            # Append the results for the episode
            all_scores.append(scores)  # Store score at end of episode
            self._test_episode_config["episode_reward"] = scores[0]

            # Append log probs for the episode to all_log_probs list
            all_log_probs.append(log_probs)

            # Log to callbacks
            if self.callbacks and not training:
                for callback in self.callbacks:
                    callback.on_test_epoch_end(epoch=step, logs=self._test_episode_config)

            print(f'Episode {episode+1}/{num_episodes} - Score: {all_scores[-1]}')

            # Reset score for this environment
            scores = 0
        
        if self.callbacks and not training:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)

        # close the environment
        env.close()

        return {
            'scores': all_scores,
            'log probs': all_log_probs,
            # 'entropy': entropy_list,
            # 'kl_divergence': kl_list
        }

    def save(self, save_dir=None):
        """
        Save the model and its configuration.

        Args:
            save_dir (str, optional): Directory to save the model. Defaults to self.save_dir.
        """
        config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of DDPG agent config
        with open(self.save_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, cls=CustomJSONEncoder)

        # saves policy and value model
        self.policy_model.save(self.save_dir)
        self.value_model.save(self.save_dir)


    @classmethod
    def load(cls, config, load_weights=True):
        """
        Load a PPO agent from a saved configuration.

        Args:
            config (dict): Configuration dictionary.
            load_weights (bool): Whether to load model weights.

        Returns:
            PPO: Loaded PPO agent.
        """

        ## create EnvSpec from config
        # env_spec = EnvSpec.from_json(config['env'])
        
        # env_wrapper = build_env_wrapper_obj(env_spec)
        env_wrapper = EnvWrapper.from_json(config["env"])


        # load policy model
        model = select_policy_model(env_wrapper)
        policy_model = model.load(config['save_dir'], load_weights)
        # load value model
        value_model = ValueModel.load(config['save_dir'], load_weights)
        # load callbacks
        callbacks = [callback_load(callback_info['class_name'], callback_info['config']) for callback_info in config['callbacks']]\
                    if config['callbacks'] else None

        # return PPO agent
        agent = cls(
            env_wrapper,
            policy_model = policy_model,
            value_model = value_model,
            discount=config["discount"],
            gae_coefficient = config["gae_coefficient"],
            policy_clip = config["policy_clip"],
            policy_clip_schedule = ScheduleWrapper(config["policy_clip_schedule"]),
            value_clip = config["value_clip"],
            value_clip_schedule = ScheduleWrapper(config["value_clip_schedule"]),
            value_loss_coefficient = config["value_loss_coefficient"],
            entropy_coefficient = config["entropy_coefficient"],
            entropy_schedule = ScheduleWrapper(config["entropy_schedule"]),
            kl_coefficient = config["kl_coefficient"],
            kl_adapter = AdaptiveKL(**config["kl_adapter"]) if config["kl_adapter"] else None,
            normalize_advantages = config["normalize_advantages"],
            normalize_values = config["normalize_values"],
            value_normalizer_clip = config["normalizer_clip"],
            policy_grad_clip = config["policy_grad_clip"],
            value_grad_clip = config["value_grad_clip"],
            reward_clip = config['reward_clip'],
            callbacks=callbacks,
            save_dir=config["save_dir"],
            device=config["device"],
        )

        # if agent.normalize_inputs:
        #     agent.state_normalizer = helper.Normalizer.load_state(config['save_dir'] + "state_normalizer.npz")

        return agent
    
    @classmethod
    def sweep_train(cls, config, env_spec, callbacks, run_number):
        """
        Train agents based on a sweep configuration.

        Args:
            config (dict): Configuration for the sweep.
            env_spec: Environment specification.
            callbacks (list): List of callbacks.
            run_number (int): Run number for logging.

        Returns:
            None
        """
        # Import necessary functions directly from wandb_support
        from wandb_support import get_wandb_config_value, get_wandb_config_optimizer_params

        logger.debug(f"init_sweep fired")
        try:
            # Instantiate env from env_spec
            env_spec = gym.envs.registration.EnvSpec.from_json(env_spec)
            env_library = config["parameters"]["env_library"]
            env_wrappers = config["parameters"]["env_wrappers"]
            if env_library == 'Gymnasium':
                env = GymnasiumWrapper(env_spec, env_wrappers)
            # env = gym.make(gym.envs.registration.EnvSpec.from_json(env_spec))

            # logger.debug(f"train config: {train_config}")
            print(f"env library: {env_library}")
            print(f"env wrappers: {env_wrappers}")
            print(f"env spec id: {env.spec.id}")
            print(f"callbacks: {callbacks}")
            print(f"run number: {run_number}")
            print(f"config set: {config}")
            agent_type = config['model_type']
            print(f"agent type: {agent_type}")

            # Get devicez
            device = get_wandb_config_value(config, agent_type, 'none', 'device')

            # Format policy and value layers, and kernels
            model_config = wandb_support.format_layers(config)

            # Policy
            # Learning Rate
            policy_learning_rate_const = get_wandb_config_value(config, agent_type, 'policy', 'learning_rate_constant')
            policy_learning_rate_exp = get_wandb_config_value(config, agent_type, 'policy', 'learning_rate_exponent')
            policy_learning_rate = policy_learning_rate_const * (10 ** policy_learning_rate_exp)
            logger.debug(f"policy learning rate set to {policy_learning_rate}")
            # Distribution
            distribution = get_wandb_config_value(config, agent_type, 'policy', 'distribution')
            # Optimizer
            policy_optimizer = get_wandb_config_value(config, agent_type, 'policy', 'optimizer')
            logger.debug(f"policy optimizer set to {policy_optimizer}")
            # Get optimizer params
            optimizer_params = get_wandb_config_optimizer_params(config, agent_type, 'policy', 'optimizer')
            policy_optimizer_params = {'type':policy_optimizer, 'params':optimizer_params}
            logger.debug(f"policy optimizer params set to {policy_optimizer_params}")
            # Get correct policy model for env action space
            if isinstance(env.action_space, gym.spaces.Discrete):
                policy_model = StochasticDiscretePolicy(
                    env = env,
                    layer_config = model_config['policy']['hidden'],
                    output_layer_kernel = model_config['policy']['output'],
                    optimizer_params = policy_optimizer_params,
                    learning_rate = policy_learning_rate,
                    distribution = distribution,
                    device = device,
                )
            # Check if the action space is continuous
            elif isinstance(env.action_space, gym.spaces.Box):
                policy_model = StochasticContinuousPolicy(
                    env = env,
                    layer_config = model_config['policy']['hidden'],
                    output_layer_kernel = model_config['policy']['output'],
                    optimizer_params = policy_optimizer_params,
                    distribution = distribution,
                    device = device,
                )
            logger.debug(f"policy model built: {policy_model.get_config()}")

            # Value Func
            # Learning Rate
            value_learning_rate_const = get_wandb_config_value(config, agent_type, 'value', "learning_rate_constant")
            value_learning_rate_exp = get_wandb_config_value(config, agent_type, 'value', "learning_rate_exponent")
            critic_learning_rate = value_learning_rate_const * (10 ** value_learning_rate_exp)
            logger.debug(f"value learning rate set to {critic_learning_rate}")
            # Optimizer
            value_optimizer = get_wandb_config_value(config, agent_type, 'value', 'optimizer')
            logger.debug(f"value optimizer set to {value_optimizer}")
            optimizer_params = get_wandb_config_optimizer_params(config, agent_type, 'value', 'optimizer')
            value_optimizer_params = {'type':value_optimizer, 'params':optimizer_params}
            logger.debug(f"value optimizer params set to {value_optimizer_params}")

            value_model = ValueModel(
                env = env,
                layer_config = model_config['value']['hidden'],
                output_layer_kernel=model_config['value']['output'],
                optimizer_params = value_optimizer_params,
                device=device,
            )
            logger.debug(f"value model built: {value_model.get_config()}")

            # Discount
            discount = get_wandb_config_value(config, agent_type, 'none', 'discount')
            # GAE coefficient
            gae_coeff = get_wandb_config_value(config, agent_type, 'none', 'advantage')
            logger.debug(f"gae coeff set to {gae_coeff}")
            # Policy clip
            policy_clip = get_wandb_config_value(config, agent_type, 'policy', 'clip_range')
            logger.debug(f"policy clip set to {policy_clip}")
            # Value clip
            value_clip = get_wandb_config_value(config, agent_type, 'value', 'clip_range')
            logger.debug(f"value clip set to {value_clip}")
            # Entropy coefficient
            entropy_coeff = get_wandb_config_value(config, agent_type, 'none', 'entropy')
            logger.debug(f"entropy coeff set to {entropy_coeff}")
            # Normalize advantages
            normalize_advantages = get_wandb_config_value(config, agent_type, 'none', 'normalize_advantage')
            logger.debug(f"normalize advantage set to {normalize_advantages}")
            # Normalize values
            normalize_values = get_wandb_config_value(config, agent_type, 'value', 'normalize_values')
            logger.debug(f"normalize values set to {normalize_values}")
            # Normalize values clip value
            normalize_val_clip = get_wandb_config_value(config, agent_type, 'value', 'normalize_values_clip')
            if normalize_val_clip == 'infinity':
                normalize_val_clip = np.inf
            logger.debug(f"normalize values clip set to {normalize_val_clip}")
            # Policy gradient clip
            policy_grad_clip = get_wandb_config_value(config, agent_type, 'policy', 'grad_clip')
            # Change value of policy_grad_clip to np.inf if == 'infinity'
            if policy_grad_clip == "infinity":
                policy_grad_clip = np.inf
            logger.debug(f"policy grad clip set to {policy_grad_clip}")
            # Value gradient clip
            value_grad_clip = get_wandb_config_value(config, agent_type, 'value', 'grad_clip')
            # Change value of policy_grad_clip to np.inf if == 'infinity'
            if value_grad_clip == "infinity":
                value_grad_clip = np.inf
            logger.debug(f"value grad clip set to {value_grad_clip}")

            # Value Loss coefficient
            value_coeff = get_wandb_config_value(config, agent_type, 'value', 'loss_coeff')
            logger.debug(f"gae coeff set to {value_coeff}")

            # Reward clip
            reward_clip = get_wandb_config_value(config, agent_type, 'none', 'reward_clip')

            # Save dir
            save_dir = get_wandb_config_value(config, agent_type, 'none', 'save_dir')
            logger.debug(f"save dir set: {save_dir}")


            # create PPO agent
            ppo_agent= PPO(
                env = env,
                policy_model = policy_model,
                value_model = value_model,
                discount = discount,
                gae_coefficient = gae_coeff,
                policy_clip = policy_clip,
                value_clip = value_clip,
                value_loss_coefficient = value_coeff,
                entropy_coefficient = entropy_coeff,
                normalize_advantages = normalize_advantages,
                normalize_values = normalize_values,
                value_normalizer_clip = normalize_val_clip,
                policy_grad_clip = policy_grad_clip,
                value_grad_clip = value_grad_clip,
                reward_clip = reward_clip,
                callbacks = callbacks,
                device = device,
            )
            logger.debug(f"PPO agent built: {ppo_agent.get_config()}")

            timesteps = get_wandb_config_value(config, agent_type, 'none', 'num_timesteps')
            traj_length = get_wandb_config_value(config, agent_type, 'none', 'trajectory_length')
            batch_size = get_wandb_config_value(config, agent_type, 'none', 'batch_size')
            learning_epochs = get_wandb_config_value(config, agent_type, 'none', 'learning_epochs')
            num_envs = get_wandb_config_value(config, agent_type, 'none', 'num_envs')
            seed = get_wandb_config_value(config, agent_type, 'none', 'seed')

            ppo_agent.train(
                timesteps = timesteps,
                trajectory_length = traj_length,
                batch_size = batch_size,
                learning_epochs = learning_epochs,
                num_envs = num_envs,
                seed = seed,
                render_freq = 0,
            )

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

    def get_config(self):
        """
        Get the current configuration of the PPO agent.

        Returns:
            dict: Configuration dictionary.
        """
        return {
                "agent_type": self.__class__.__name__,
                # "env": serialize_env_spec(self.env.spec),
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
                "normalize_values": self.normalize_values,
                "normalizer_clip": self.value_norm_clip,
                "policy_grad_clip": self.policy_grad_clip,
                "value_grad_clip": self.value_grad_clip,
                "reward_clip": self.reward_clip,
                "callbacks": [callback.get_config() for callback in self.callbacks] if self.callbacks else None,
                "save_dir": self.save_dir,
                "device": self.device,
                # "seed": self.seed,
            }

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

