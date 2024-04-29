"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
import json
import os
from typing import List
from pathlib import Path
import time
import datetime
import inspect
import threading
from mpi4py import MPI
from helper import MPIHelper

import rl_callbacks
import models
import cnn_models
import wandb
import wandb_support
import helper
import dash_callbacks
import gym_helper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np






# Agent class
class Agent:
    """Base class for all RL agents."""

    def get_action(self, state):
        """Returns an action given a state."""

    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None
    ):
        """Trains the model for 'episodes' number of episodes."""

    def learn(self):
        """Updates the model."""

    def test(self, num_episodes=None, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""

    def save(self):
        """Saves the model."""

    @classmethod
    def load(cls, folder: str = "models"):
        """Loads the model."""


class ActorCritic(Agent):
    """Actor Critic Agent."""

    def __init__(
        self,
        env: gym.Env,
        policy_model: models.PolicyModel,
        value_model: models.ValueModel,
        discount=0.99,
        policy_trace_decay: float = 0.0,
        value_trace_decay: float = 0.0,
        callbacks: list = [],
        save_dir: str = "models/",
    ):
        
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.discount = discount
        self.policy_trace_decay = policy_trace_decay
        self.value_trace_decay = value_trace_decay
        self.callbacks = callbacks
        self.save_dir = save_dir

        if callbacks:
            self._train_config = {}
            self._train_episode_config = {}
            self._train_step_config = {}
            self._test_config = {}
            self._test_episode_config = {}

            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, rl_callbacks.WandbCallback):  
                    self._wandb = True

        else:
            self.callbacks = []
            self._wandb = False

        self._step = None
        # set self.action to None
        self.action = None
        # instantiate and set policy and value traces
        self.policy_trace = []
        self.value_trace = []
        self._set_traces()

    @classmethod
    def build(
        cls,
        env,
        policy_layers,
        value_layers,
        callbacks,
        config,#: wandb.config,
        save_dir: str = "models/",
    ):
        """Builds the agent."""
        policy_optimizer = wandb.config.policy_optimizer
        value_optimizer = wandb.config.value_optimizer
        policy_learning_rate = wandb.config.learning_rate
        value_learning_rate = wandb.config.learning_rate
        policy_model = models.PolicyModel(
            env, dense_layers=policy_layers, optimizer=policy_optimizer, learning_rate=policy_learning_rate
        )
        value_model = models.ValueModel(
            env, dense_layers=value_layers, optimizer=value_optimizer, learning_rate=value_learning_rate
        )

        return cls(
            env,
            policy_model,
            value_model,
            config.discount,
            config.policy_trace_decay,
            config.value_trace_decay,
            callbacks,
            save_dir=save_dir,
        )

    def _initialize_env(self, render=False, render_freq=10, context=None):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            if context == "train":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/training",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )
            elif context == "test":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/testing",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )

        return gym.make(self.env.spec)

    def _set_traces(self):
        for weights in self.policy_model.parameters():
            self.policy_trace.append(torch.zeros_like(weights, device=self.device))
            #DEBUG
            print(f'policy trace shape: {weights.size()}')

        for weights in self.value_model.parameters():
            self.value_trace.append(torch.zeros_like(weights, device=self.device))
            #DEBUG
            print(f'value trace shape: {weights.size()}')


    def _update_traces(self):
        with torch.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                self.policy_trace[i] = (
                    self.discount * self.policy_trace_decay * self.policy_trace[i]
                ) + (self.influence * weights.grad)

            for i, weights in enumerate(self.value_model.parameters()):
                self.value_trace[i] = (self.discount * self.value_trace_decay * self.value_trace[i]) + weights.grad

        # log to train step
        for i, (v_trace, p_trace) in enumerate(zip(self.value_trace, self.policy_trace)):
            self._train_step_config[f"value trace {i}"] = torch.histc(v_trace, bins=20)
            self._train_step_config[f"policy trace {i}"] = torch.histc(p_trace, bins=20)

    def get_action(self, state):
        state =  torch.from_numpy(state).to(self.device)
        logits = self.policy_model(state)
        self.probabilities = F.softmax(logits, dim=-1)
        probabilities_dist = Categorical(probs=self.probabilities)
        action = probabilities_dist.sample()
        self.log_prob = probabilities_dist.log_prob(action)
        
        return action.item()

    def train(self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
        """Trains the model for 'episodes' number of episodes."""
        # set models to train mode
        self.policy_model.train()
        self.value_model.train()
        # set save_dir if not None
        if save_dir:
            self.save_dir = save_dir
        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, rl_callbacks.WandbCallback):
                    callback.on_train_begin((self.policy_model, self.value_model,), logs=self._config)

                else:
                    callback.on_train_begin(logs=self._config)
        reward_history = []
        self.env = self._initialize_env(render, render_freq, 'train')
        # set step counter
        self._step = 1
        # set best reward
        best_reward = self.env.reward_range[0]
        
        for i in range(num_episodes):
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            self.influence = 1
            while not done:
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_begin(step=self._step, logs=None)
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                if term or trunc:
                    done = True
                episode_reward += reward
                self.learn(state, reward, next_state, done)
                if self._wandb:
                    self._train_step_config["action"] = action
                    self._train_step_config["step reward"] = reward
                    self._train_step_config["influence"] = self.influence

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_end(step=self._step, logs=self._train_step_config)
                state = next_state
                self.influence *= self.discount
                self._step += 1
            self._train_episode_config["episode"] = i+1
            reward_history.append(episode_reward)
            self._train_episode_config["episode reward"] = episode_reward
            avg_reward = np.mean(reward_history[-100:])
            self._train_episode_config["avg reward"] = avg_reward
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

            print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

        # close the environment
        self.env.close()
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)

    def learn(self, state, reward, next_state, done):
        self.policy_model.optimizer.zero_grad()
        self.value_model.optimizer.zero_grad()

        state = torch.tensor(state, device=self.device)
        reward = torch.tensor(reward, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        done = torch.tensor(done, device=self.device, dtype=torch.float32)

        state_value = self.value_model(state)
        next_state_value = self.value_model(next_state)
        temporal_difference = (
            reward + self.discount * next_state_value * (1 - done) - state_value)
        value_loss = temporal_difference.square()

        value_loss.backward(retain_graph=True)

        policy_loss = (-self.log_prob * temporal_difference).mean()
        policy_loss.backward()

        self._update_traces()

        #copy traces to weight gradients
        with torch.no_grad():
            for i, weights in enumerate(self.policy_model.parameters()):
                weights.grad = self.policy_trace[i]

            for i, weights in enumerate(self.value_model.parameters()):
                weights.grad = self.value_trace[i]

        self.value_model.optimizer.step()
        self.policy_model.optimizer.step()

        
        if self._wandb:
            self._train_step_config["temporal difference"] = temporal_difference
            for i, probability in enumerate(self.probabilities.squeeze()):
                self._train_step_config[f"probability action {i}"] = probability.item()
            self._train_step_config["log probability"] = self.log_prob.item()
            self._train_step_config["policy loss"] = policy_loss.item()
            self._train_step_config["value loss"] = value_loss.item()


    def test(self, num_episodes, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""
        # set models to eval mode
        self.policy_model.eval()
        self.value_model.eval()
        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, 'test')
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_begin(logs=self._config)

        self._step = 1
        for i in range(num_episodes):
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_begin(step=self._step, logs=None)
            states = []
            next_states = []
            actions = []
            rewards = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                # store trajectories
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                self._step += 1
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])
            self._test_episode_config["episode reward"] = episode_reward
            self._test_episode_config["avg reward"] = avg_reward
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(step=self._step, logs=self._test_episode_config)

            print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)
        # close the environment
        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "discount": self.discount,
            "policy_trace_decay": self.policy_trace_decay,
            "value_trace_decay": self.value_trace_decay,
            "callbacks": [callback.__class__.__name__ for callback in (self.callbacks or []) if callback is not None],
            "save_dir": self.save_dir
        }


    def save(self):
        """Saves the model."""
        obj_config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of actor critic agent config
        with open(self.save_dir + "/obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

        # saves policy and value model
        self.policy_model.save(self.save_dir)
        self.value_model.save(self.save_dir)

        # # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")

    @classmethod
    def load(cls, folder: str = "models"):
        """Loads the model."""
        # load reinforce agent config
        with open(
            Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
        ) as f:
            obj_config = json.load(f)

        # load policy model
        policy_model = models.PolicyModel.load(folder)
        # load value model
        value_model = models.ValueModel.load(folder)
        # load callbacks
        callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in obj_config['callbacks']]

        # return Actor-Critic agent
        agent = cls(
            gym.make(obj_config["env"]),
            policy_model=policy_model,
            value_model=value_model,
            discount=obj_config["discount"],
            policy_trace_decay=obj_config["policy_trace_decay"],
            value_trace_decay=obj_config["value_trace_decay"],
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )

        return agent


class Reinforce(Agent):
    def __init__(
        self,
        env: gym.Env,
        policy_model: models.PolicyModel,
        value_model: models.ValueModel = None,
        discount=0.99,
        callbacks: List = [],
        save_dir: str = "models/",
    ):
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.discount = discount
        self.callbacks = callbacks
        self.save_dir = save_dir
        
        if callbacks:
            self._train_config = {}
            self._train_episode_config = {}
            self._train_step_config = {}
            self._test_config = {}
            self._test_episode_config = {}
            
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, rl_callbacks.WandbCallback):  
                    self._wandb = True

        else:
            self.callbacks = []
            self._wandb = False

        self._step = None
        self._cur_learning_steps = None

    @classmethod
    def build(
        cls,
        env,
        policy_layers,
        value_layers,
        callbacks,
        config,#: wandb.config,
        save_dir: str = "models/",
    ):
        """Builds the agent."""
        policy_optimizer = config.policy_optimizer
        value_optimizer = config.value_optimizer
        policy_model = models.PolicyModel(
            env, dense_layers=policy_layers, optimizer=policy_optimizer, learning_rate=config.learning_rate
        )
        value_model = models.ValueModel(
            env, dense_layers=value_layers, optimizer=value_optimizer, learning_rate=config.learning_rate
        )

        return cls(
            env,
            policy_model,
            value_model,
            config.discount,
            callbacks,
            save_dir=save_dir,
        )


    def _initialize_env(self, render=False, render_freq=10, context=None):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            if context == "train":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/training",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )
            elif context == "test":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/testing",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )

        return gym.make(self.env.spec)

    def get_return(self, rewards):
        """Compute expected returns per timestep."""

        returns = []
        discounted_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for reward in reversed(rewards):
            discounted_sum = reward + self.discount * discounted_sum
            returns.append(discounted_sum)
        return torch.tensor(returns[::-1], dtype=torch.float32, device=self.device)

    def get_action(self, state):
        state =  torch.from_numpy(state).to(self.device)
        logits = self.policy_model(state)
        probabilities = F.softmax(logits, dim=-1)
        probabilities_dist = Categorical(probs=probabilities)
        action = probabilities_dist.sample()
        log_prob = probabilities_dist.log_prob(action).unsqueeze(0)

        for i, probability in enumerate(probabilities.squeeze()):
            self._train_step_config[f"probability action {i}"] = probability
        
        return action, log_prob

    def learn(self, states, log_probs, rewards):
        # Clear gradients
        self.policy_model.optimizer.zero_grad()
        self.value_model.optimizer.zero_grad()

        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        returns = self.get_return(rewards)
        
        policy_loss = 0
        value_loss = 0
        
        for t, (state, log_prob, _return, reward, step) in enumerate(
            zip(states, log_probs, returns, rewards, self._cur_learning_steps)
        ):
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_step_begin(step=step, logs=None)
            
            # Calculate step loss for value model if present
            if self.value_model:
                state_value = self.value_model(state)
                # calculate performance
                advantage = _return - state_value
                # calculate value loss
                value_loss += advantage**2

            policy_loss += -log_prob * advantage * self.discount**t

            # log to wandb if using wandb callback
            if self._wandb:
                self._train_step_config["temporal difference"] = advantage
                self._train_step_config["policy loss"] = policy_loss
                self._train_step_config["value loss"] = value_loss


        # Calculate gradients
        policy_loss = policy_loss
        value_loss = value_loss

        policy_loss.backward(retain_graph=True)
        value_loss.backward()

        # Update weights
        self.policy_model.optimizer.step()
        self.value_model.optimizer.step()


    def train(self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
        """Trains the model for 'episodes' number of episodes."""

        # set models to train mode
        self.policy_model.train()
        self.value_model.train()

        # set save dir if provided
        if save_dir:
            self.save_dir = save_dir

        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, rl_callbacks.WandbCallback):
                    callback.on_train_begin((self.policy_model, self.value_model,), logs=self._config)
        
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, 'train')
        # set step counter
        self._step = 0
        # set current learning steps
        self._cur_learning_steps = []
        # set best reward
        best_reward = self.env.reward_range[0]
        # instantiate list to store reward history
        reward_history = []
        for i in range(num_episodes):
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            states = []
            next_states = []
            # actions = []
            log_probs = []
            rewards = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_begin(step=self._step, logs=None)
                self._cur_learning_steps.append(self._step)
                action, log_prob = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                # store trajectories
                states.append(state)
                # actions.append(action)
                log_probs.append(log_prob)
                next_states.append(next_state)
                rewards.append(reward)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                
                # log to train log
                self._train_step_config["action"] = action
                self._train_step_config["step reward"] = reward
                self._train_step_config["log probability"] = log_prob

                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_end(step=self._step, logs=self._train_step_config)
                if not done:
                    self._step += 1
                
            self.learn(states, log_probs, rewards)
            # clear learning steps array
            self._cur_learning_steps = []
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])

            self._train_episode_config["episode"] = i
            self._train_episode_config["episode reward"] = episode_reward
            self._train_episode_config["avg reward"] = avg_reward
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_end(
                    epoch=self._step, logs=self._train_episode_config
                )

            print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)
        # close the environment
        self.env.close()

    def test(self, num_episodes, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""

        # Set models to eval mode
        self.policy_model.eval()
        self.value_model.eval()
        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, 'test')
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_begin(logs=None)
        self._step = 1
        for i in range(num_episodes):
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_begin(step=self._step, logs=None)
            states = []
            next_states = []
            actions = []
            rewards = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                # store trajectories
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                self._step += 1
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])
            self._test_episode_config["episode reward"] = episode_reward
            self._test_episode_config["avg reward"] = avg_reward
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_step_end(
                    step=self._step, logs=self._test_episode_config
                )

            print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_end(logs=self._test_episode_config)
        # close the environment
        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "discount": self.discount,
            "callbacks": [callback.get_config() for callback in self.callbacks if self.callbacks is not None],
            "save_dir": self.save_dir
        }

    def save(self):
        """Saves the model."""
        obj_config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of reinforce agent config
        with open(self.save_dir + "/obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

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
    def load(cls, folder: str = "models"):
        """Loads the model."""
        # load reinforce agent config
        with open(
            Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
        ) as f:
            obj_config = json.load(f)

        # load policy model
        policy_model = models.PolicyModel.load(folder)
        # load value model
        value_model = models.ValueModel.load(folder)
        # load callbacks
        callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in obj_config['callbacks']]

        # return reinforce agent
        agent = cls(
            gym.make(obj_config["env"]),
            policy_model=policy_model,
            value_model=value_model,
            discount=obj_config["discount"],
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )

        return agent
    

class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        env: gym.Env,
        actor_model: models.ActorModel,
        critic_model: models.CriticModel,
        discount=0.99,
        tau=0.001,
        action_epsilon: float = 0.0,
        replay_buffer: helper.ReplayBuffer = None,
        batch_size: int = 64,
        noise = None,
        normalize_inputs: bool = False,
        normalize_kwargs: dict = {},
        callbacks: List = [],
        save_dir: str = "models/ddpg/",
        _DEBUG: bool = False
    ):
        self.env = env
        self.actor_model = actor_model
        self.critic_model = critic_model
        # set target actor and critic models
        self.target_actor_model = self.clone_model(self.actor_model)
        self.target_critic_model = self.clone_model(self.critic_model)
        self.discount = discount
        self.tau = tau
        self.action_epsilon = action_epsilon
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.noise = noise
        self.normalize_inputs = normalize_inputs
        self.normalize_kwargs = normalize_kwargs
        
        # set internal attributes
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            self._obs_space_shape = env.observation_space['observation'].shape
        else:
            self._obs_space_shape = env.observation_space.shape

        if self.normalize_inputs:
            self.state_normalizer = helper.Normalizer(size=self._obs_space_shape, **self.normalize_kwargs)
        
        self.save_dir = save_dir
        self._DEBUG = _DEBUG

        # instantiate internal attribute use_her to be switched by HER class if using DDPG
        self._use_her = False
        
        self.callbacks = callbacks
        if callbacks:
            for callback in self.callbacks:
                self._config = callback._config(self)
                if isinstance(callback, rl_callbacks.WandbCallback):  
                    self._wandb = True

        else:
            self.callback_list = None
            self._wandb = False
        self._train_config = {}
        self._train_episode_config = {}
        self._train_step_config = {}
        self._test_config = {}
        self._test_episode_config = {}

        self._step = None

    def clone(self):
        env = gym.make(self.env.spec)
        actor = self.clone_model(self.actor_model)
        critic = self.clone_model(self.critic_model)
        replay_buffer = self.replay_buffer.clone()
        noise = self.noise.clone()

        return DDPG(
            env,
            actor,
            critic,
            self.discount,
            self.tau,
            self.action_epsilon,
            replay_buffer,
            self.batch_size,
            noise,
            self.normalize_inputs,
            self.normalize_kwargs,

        )
        
    
    def clone_model(self, model):
        """Clones a model."""
        return model.get_clone()
    
    @classmethod
    def build(
        cls,
        env,
        actor_cnn_layers,
        critic_cnn_layers,
        actor_layers,
        critic_state_layers,
        critic_merged_layers,
        callbacks,
        config,#: wandb.config,
        save_dir: str = "models/",
    ):
        """Builds the agent."""
        # Actor
        actor_learning_rate=config[config.model_type][f"{config.model_type}_actor_learning_rate"]
        actor_optimizer = config[config.model_type][f"{config.model_type}_actor_optimizer"]
        # get optimizer params
        actor_optimizer_params = {}
        if actor_optimizer == "Adam":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
        
        elif actor_optimizer == "Adagrad":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            actor_optimizer_params['lr_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
        
        elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            actor_optimizer_params['momentum'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']

        actor_normalize_layers = config[config.model_type][f"{config.model_type}_actor_normalize_layers"]

        # Critic
        critic_learning_rate=config[config.model_type][f"{config.model_type}_critic_learning_rate"]
        critic_optimizer = config[config.model_type][f"{config.model_type}_critic_optimizer"]
        critic_optimizer_params = {}
        if critic_optimizer == "Adam":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
        
        elif critic_optimizer == "Adagrad":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            critic_optimizer_params['lr_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
        
        elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            critic_optimizer_params['momentum'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
        
        critic_normalize_layers = config[config.model_type][f"{config.model_type}_critic_normalize_layers"]

        # Check if CNN layers and if so, build CNN model
        if actor_cnn_layers:
            actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
        else:
            actor_cnn_model = None

        if critic_cnn_layers:
            critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
        else:
            critic_cnn_model = None
        
        actor_model = models.ActorModel(env = env,
                                        cnn_model = actor_cnn_model,
                                        dense_layers = actor_layers,
                                        optimizer = actor_optimizer,
                                        optimizer_params = actor_optimizer_params,
                                        learning_rate = actor_learning_rate,
                                        normalize_layers = actor_normalize_layers
        )
        critic_model = models.CriticModel(env = env,
                                          cnn_model = critic_cnn_model,
                                          state_layers = critic_state_layers,
                                          merged_layers = critic_merged_layers,
                                          optimizer = critic_optimizer,
                                          optimizer_params = critic_optimizer_params,
                                          learning_rate = critic_learning_rate,
                                          normalize_layers = critic_normalize_layers
        )

        # action epsilon
        action_epsilon = config[config.model_type][f"{config.model_type}_epsilon_greedy"]

        # normalize inputs
        normalize_inputs = config[config.model_type][f"{config.model_type}_normalize_input"]
        normalize_kwargs = {}
        if "True" in normalize_inputs:
            normalize_kwargs = config[config.model_type][f"{config.model_type}_normalize_clip"]

        return cls(
            env = env,
            actor_model = actor_model,
            critic_model = critic_model,
            discount = config[config.model_type][f"{config.model_type}_discount"],
            tau = config[config.model_type][f"{config.model_type}_tau"],
            action_epsilon = action_epsilon,
            replay_buffer = helper.ReplayBuffer(env=env),
            batch_size = config[config.model_type][f"{config.model_type}_batch_size"],
            noise = helper.Noise.create_instance(config[config.model_type][f"{config.model_type}_noise"], shape=env.action_space.shape, **config[config.model_type][f"{config.model_type}_noise_{config[config.model_type][f'{config.model_type}_noise']}"]),
            normalize_inputs = normalize_inputs,
            normalize_kwargs = normalize_kwargs,
            callbacks = callbacks,
            save_dir = save_dir,
        )

    def _init_her(self, goal_shape, eps=None, clip_range=None):
            self.normalize_inputs = True
            self._use_her = True
            self.state_normalizer = helper.Normalizer(size=self._obs_space_shape, eps=eps, clip_range=clip_range)
            self.goal_normalizer = helper.Normalizer(size=goal_shape, eps=eps, clip_range=clip_range)
            # set clamp for targets
            self.target_clamp = 1 / (1 - self.discount)

    def _initialize_env(self, render=False, render_freq=10, context=None):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            if context == "train":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/training",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )
            elif context == "test":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/testing",
                    episode_trigger=lambda episode_id: (episode_id+1) % render_freq == 0,
                )

        return gym.make(self.env.spec)
    
    def get_action(self, state, goal=None, grad=True, test=False):

        # check if get action is for testing
        if test:
            with torch.no_grad():
                # normalize state if self.normalize_inputs
                if self.normalize_inputs:
                    state = self.state_normalizer.normalize(state)

                # make sure state is a tensor and on correct device
                state = torch.tensor(state, dtype=torch.float32, device=self.actor_model.device)
                # normalize goal if self._use_her
                if self._use_her:
                    goal = self.goal_normalizer.normalize(goal)
                    # make sure goal is a tensor and on correct device
                    goal = torch.tensor(goal, dtype=torch.float32, device=self.actor_model.device)
                
                # permute state to (C,H,W) if actor using cnn model
                if self.actor_model.cnn_model:
                    state = state.permute(2, 0, 1).unsqueeze(0)

                # get action
                _, action = self.actor_model(state, goal)
                # transfer action to cpu, detach from any graphs, tranform to numpy, and flatten
                action_np = action.cpu().detach().numpy().flatten()
        
        # check if using epsilon greedy
        else: #self.action_epsilon > 0.0:
            # if random number is less than epsilon, sample random action
            if np.random.random() < self.action_epsilon:
                action_np = self.env.action_space.sample()
                noise_np = np.zeros_like(action_np)
            
            else:
                # if gradient tracking is true
                if grad:
                    # normalize state if self.normalize_inputs
                    if self.normalize_inputs:
                        state = self.state_normalizer.normalize(state)
                        # print(f'normalized state: {state}')
                    
                    # make sure state is a tensor and on correct device
                    state = torch.tensor(state, dtype=torch.float32, device=self.actor_model.device)
                    
                    # normalize goal if self._use_her
                    if self._use_her:
                        goal = self.goal_normalizer.normalize(goal)
                        # print(f'normalized goal: {goal}')
                        # make sure goal is a tensor and on correct device
                        goal = torch.tensor(goal, dtype=torch.float32, device=self.actor_model.device)

                    # permute state to (C,H,W) if actor using cnn model
                    if self.actor_model.cnn_model:
                        state = state.permute(2, 0, 1).unsqueeze(0)

                    _, pi = self.actor_model(state, goal)
                    # print(f'pi: {pi}')
                    noise = self.noise()
                    # print(f'noise: {noise}')

                    # Convert the action space bounds to a tensor on the same device
                    action_space_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.actor_model.device)
                    action_space_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.actor_model.device)

                    action = (pi + noise).clip(action_space_low, action_space_high)
                    # print(f'action + noise: {action}')

                    noise_np = noise.cpu().detach().numpy().flatten()
                    action_np = action.cpu().detach().numpy().flatten()

                else:
                    with torch.no_grad():
                        # normalize state if self.normalize_inputs
                        if self.normalize_inputs:
                            state = self.state_normalizer.normalize(state)

                        # make sure state is a tensor and on correct device
                        state = torch.tensor(state, dtype=torch.float32, device=self.actor_model.device)
                        # normalize goal if self._use_her
                        if self._use_her:
                            goal = self.goal_normalizer.normalize(goal)
                            # make sure goal is a tensor and on correct device
                            goal = torch.tensor(goal, dtype=torch.float32, device=self.actor_model.device)
                        
                        # permute state to (C,H,W) if actor using cnn model
                        if self.actor_model.cnn_model:
                            state = state.permute(2, 0, 1).unsqueeze(0)

                        _, pi = self.actor_model(state, goal)
                        noise = self.noise()

                        # Convert the action space bounds to a tensor on the same device
                        action_space_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.actor_model.device)
                        action_space_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.actor_model.device)

                        action = (pi + noise).clip(action_space_low, action_space_high)

                        noise_np = noise.cpu().detach().numpy().flatten()
                        action_np = action.cpu().detach().numpy().flatten()

        if test:
            # loop over all actions to log to wandb
            for i, a in enumerate(action_np):
                # Log the values to wandb
                self._train_step_config[f'action_{i}'] = a

        else:
            # Loop over the noise and action values and log them to wandb
            for i, (a,n) in enumerate(zip(action_np, noise_np)):
                # Log the values to wandb
                self._train_step_config[f'action_{i}'] = a
                self._train_step_config[f'noise_{i}'] = n

        return action_np


    def learn(self):
        
        # with torch.no_grad():
            # sample a batch of experiences from the replay buffer
            # if using HER
        if self._use_her:
            states, actions, rewards, next_states, dones, achieved_goals, next_achieved_goals, desired_goals = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # if normalize states if self.normalize_inputs
        if self.normalize_inputs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        
        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.actor_model.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.actor_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.actor_model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.actor_model.device)
        dones = torch.tensor(dones, dtype=torch.int8, device=self.actor_model.device)

        # if using HER, normalize goals and convert to tensors
        if self._use_her:
            desired_goals = self.goal_normalizer.normalize(desired_goals)
            # convert desired goals to tensor and place on correct device
            desired_goals = torch.tensor(desired_goals, dtype=torch.float32, device=self.actor_model.device)
        else:
            # set desired goals to None
            desired_goals = None

        # permute states and next states if using cnn
        if self.actor_model.cnn_model:
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)

        # convert rewards and dones to 2d tensors
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # check shapes
        # print(f'states shape: {states.shape}')
        # print(f'action shape: {actions.shape}')
        # print(f'rewards shape: {rewards.shape}')
        # print(f'next states shape: {next_states.shape}')
        # print(f'desired goals shape: {desired_goals.shape}')

        # get target values 
        _, target_actions = self.target_actor_model(next_states, desired_goals)
        # print(f'target actions shape: {target_actions.shape}')
        # print(f'target actions: {target_actions}')
        target_critic_values = self.target_critic_model(next_states, target_actions, desired_goals)
        # print(f'target critic values shape: {target_critic_values.shape}')
        # print(f'target critic values: {target_critic_values}')
        targets = rewards + self.discount * target_critic_values# * (1 - dones)
        # print(f'targets shape: {targets.shape}')
        # print(f'targets: {targets}')
        if self._use_her:
            targets = torch.clamp(targets, min=-self.target_clamp, max=self.target_clamp)
            # print(f'clamped targets: {targets}')
        
        
        # get current critic values and calculate critic loss
        prediction = self.critic_model(states, actions, desired_goals)
        # print(f'prediction shape: {prediction.shape}')
        # print(f'predictions: {prediction}')
        critic_loss = F.mse_loss(prediction, targets)
        # print(f'critic loss shape: {critic_loss.shape}')
        # print(f'critic loss: {critic_loss}')
        
        # update critic
        self.critic_model.optimizer.zero_grad()
        critic_loss.backward()
        if self._use_her:
            helper.sync_grads(self.critic_model)
        self.critic_model.optimizer.step()
        
        
        # update actor
        pre_act_values, action_values = self.actor_model(states, desired_goals)
        # print(f'pre act values shape: {pre_act_values.shape}')
        # print(f'pre act values: {pre_act_values}')
        # print(f'action values shape: {action_values.shape}')
        # print(f'action values: {action_values}')
        critic_values = self.critic_model(states, action_values, desired_goals)
        # print(f'critic values shape: {critic_values.shape}')
        # print(f'critic values: {critic_values}')
        actor_loss = -critic_values.mean()
        # print(f'actor loss shape: {actor_loss.shape}')
        # print(f'actor loss: {actor_loss}')
        if self._use_her:
            actor_loss += (pre_act_values ** 2).mean()
            # print(f'actor loss her shape: {actor_loss.shape}')
            # print(f'actor loss her: {actor_loss}')
        
        self.actor_model.optimizer.zero_grad()
        actor_loss.backward()
        if self._use_her:
            helper.sync_grads(self.actor_model)
        self.actor_model.optimizer.step()

        # add metrics to step_logs
        self._train_step_config['actor predictions'] = action_values.mean()
        self._train_step_config['critic predictions'] = critic_values.mean()
        self._train_step_config['target actor predictions'] = target_actions.mean()
        self._train_step_config['target critic predictions'] = target_critic_values.mean()
        
        return actor_loss.item(), critic_loss.item()
        
    
    def soft_update(self, current, target):
        with torch.no_grad():
            for current_params, target_params in zip(current.parameters(), target.parameters()):
                target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)


    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
        """Trains the model for 'episodes' number of episodes."""

        # set models to train mode
        self.actor_model.train()
        self.critic_model.train()

        if save_dir:
            self.save_dir = save_dir
        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, rl_callbacks.WandbCallback):
                    callback.on_train_begin((self.critic_model, self.actor_model,), logs=self._config)

                else:
                    callback.on_train_begin(logs=self._config)

        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, context='train')
        
        # initialize step counter (for logging)
        self._step = 1
        # set best reward
        best_reward = self.env.reward_range[0]
        # instantiate list to store reward history
        reward_history = []
        # instantiate lists to store time history
        episode_time_history = []
        step_time_history = []
        learning_time_history = []
        steps_per_episode_history = []  # List to store steps per episode
        for i in range(num_episodes):
            episode_start_time = time.time()
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_begin(epoch=self._step, logs=None)
            # reset noise
            if type(self.noise) == helper.OUNoise:
                self.noise.reset()
            # reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0  # Initialize steps counter for the episode
            while not done:
                # run callbacks on train batch begin
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_begin(step=self._step, logs=None)
                step_start_time = time.time()
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                # extract observation from next state if next_state is dict (robotics)
                if isinstance(next_state, dict):
                    next_state = next_state['observation']
                step_time = time.time() - step_start_time
                step_time_history.append(step_time)
                # store trajectory in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                episode_steps += 1
                
                # check if enough samples in replay buffer and if so, learn from experiences
                if self.replay_buffer.counter > self.batch_size:
                    learn_time = time.time()
                    actor_loss, critic_loss = self.learn()
                    self._train_step_config["actor loss"] = actor_loss
                    self._train_step_config["critic loss"] = critic_loss
                    # perform soft update on target networks
                    self.soft_update(self.actor_model, self.target_actor_model)
                    self.soft_update(self.critic_model, self.target_critic_model)

                    learning_time_history.append(time.time() - learn_time)

                self._train_step_config["step reward"] = reward
                self._train_step_config["step time"] = step_time
                
                # log to wandb if using wandb callback
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_train_step_end(step=self._step, logs=self._train_step_config)
                if not done:
                    self._step += 1
            
            episode_time = time.time() - episode_start_time
            episode_time_history.append(episode_time)
            reward_history.append(episode_reward)
            steps_per_episode_history.append(episode_steps) 
            avg_reward = np.mean(reward_history[-100:])
            avg_episode_time = np.mean(episode_time_history[-100:])
            avg_step_time = np.mean(step_time_history[-100:])
            avg_learn_time = np.mean(learning_time_history[-100:])
            avg_steps_per_episode = np.mean(steps_per_episode_history[-100:])  # Calculate average steps per episode

            self._train_episode_config['episode'] = i
            self._train_episode_config["episode reward"] = episode_reward
            self._train_episode_config["avg reward"] = avg_reward
            self._train_episode_config["episode time"] = episode_time

            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_train_epoch_end(epoch=self._step, logs=self._train_episode_config)

            print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}, episode_time {episode_time:.2f}s, avg_episode_time {avg_episode_time:.2f}s, avg_step_time {avg_step_time:.6f}s, avg_learn_time {avg_learn_time:.6f}s, avg_steps_per_episode {avg_steps_per_episode:.2f}")

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_end(logs=self._train_episode_config)
        # close the environment
        self.env.close()

       
    def test(self, num_episodes, render, render_freq):
        """Runs a test over 'num_episodes'."""

        # set model in eval mode
        self.actor_model.eval()
        self.critic_model.eval()

        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, context='test')
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_test_begin(logs=self._config)

        self._step = 1
        # set the model to calculate no gradients during evaluation
        with torch.no_grad():
            for i in range(num_episodes):
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_begin(epoch=self._step, logs=None) # update to pass any logs if needed
                states = []
                next_states = []
                actions = []
                rewards = []
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.get_action(state, test=True)
                    next_state, reward, term, trunc, _ = self.env.step(action.numpy(force=True))
                    # extract observation from next state if next_state is dict (robotics)
                    if isinstance(next_state, dict):
                        next_state = next_state['observation']
                    # store trajectories
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    if term or trunc:
                        done = True
                    episode_reward += reward
                    state = next_state
                    self._step += 1
                reward_history.append(episode_reward)
                avg_reward = np.mean(reward_history[-100:])
                self._test_episode_config["episode_reward"] = episode_reward
                self._test_episode_config["avg_reward"] = avg_reward
                if self.callbacks:
                    for callback in self.callbacks:
                        callback.on_test_epoch_end(epoch=self._step, logs=self._test_episode_config)

                print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_test_end(logs=self._test_episode_config)
            # close the environment
            self.env.close()


    def get_config(self):
        return {
                "agent_type": self.__class__.__name__,
                "env": self.env.spec.id,
                "actor_model": self.actor_model.get_config(),
                "critic_model": self.critic_model.get_config(),
                "discount": self.discount,
                "tau": self.tau,
                "action_epsilon": self.action_epsilon,
                "replay_buffer": self.replay_buffer.get_config(),
                "batch_size": self.batch_size,
                "noise": self.noise.get_config(),
                'normalize_inputs': self.normalize_inputs,
                'normalize_kwargs': self.normalize_kwargs,
                "callbacks": [callback.get_config() for callback in self.callbacks if self.callbacks is not None],
                "save_dir": self.save_dir
            }


    def save(self):
        """Saves the model."""
        obj_config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of DDPG agent config
        with open(self.save_dir + "/obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

        # saves policy and value model
        self.actor_model.save(self.save_dir)
        self.critic_model.save(self.save_dir)

        if self.normalize_inputs:
            self.state_normalizer.save_state(self.save_dir + "/state_normalizer.npz")

        # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")


    @classmethod
    def load(cls, folder: str = "models/ddpg", load_weights=True):
        """Loads the model."""
        # load reinforce agent config
        with open(
            Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
        ) as f:
            obj_config = json.load(f)

        # load policy model
        actor_model = models.ActorModel.load(folder)
        # load value model
        critic_model = models.CriticModel.load(folder)
        # load replay buffer
        obj_config['replay_buffer']['config']['env'] = gym.make(obj_config['env'])
        replay_buffer = helper.ReplayBuffer(**obj_config["replay_buffer"]["config"])
        # load noise
        noise = helper.Noise.create_instance(obj_config["noise"]["class_name"], **obj_config["noise"]["config"])
        # if normalizer, load
        normalize_inputs = obj_config['normalize_inputs']
        normalize_kwargs = obj_config['normalize_kwargs']
        # load callbacks
        callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in obj_config['callbacks']]

        # return DDPG agent
        agent = cls(
            gym.make(obj_config["env"]),
            actor_model = actor_model,
            critic_model = critic_model,
            discount=obj_config["discount"],
            tau=obj_config["tau"],
            action_epsilon=obj_config["action_epsilon"],
            replay_buffer=replay_buffer,
            batch_size=obj_config["batch_size"],
            noise=noise,
            normalize_inputs = normalize_inputs,
            normalize_kwargs = normalize_kwargs,
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )

        if agent.normalize_inputs:
            agent.state_normalizer = helper.Normalizer.load_state(folder + "/state_normalizer.npz")
        
        # if callbacks:
        #     for callback in callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             agent._wandb = True
        #             agent._train_config = {}
        #             agent._train_episode_config = {}
        #             agent._train_step_config = {}
        #             agent._test_config = {}

        return agent


class HER(Agent):

    def __init__(self, agent:Agent, strategy:str='final', tolerance:float=0.0, num_goals:int=4, desired_goal:callable=None,
                 achieved_goal:callable=None, reward_fn:callable=None, normalizer_clip:float=5.0,
                 normalizer_eps:float=0.01, save_dir: str = "models/her/"):
        super().__init__()
        self.agent = agent
        self.strategy = strategy
        self.tolerance = tolerance
        self.num_goals = num_goals
        self.desired_goal_func = desired_goal
        self.achieved_goal_func = achieved_goal
        self.reward_fn = reward_fn
        self.normalizer_clip = normalizer_clip
        self.normalizer_eps = normalizer_eps
        self.save_dir = save_dir
        # self.lock = threading.Lock()
        self.mpi_helper = MPIHelper()
        self.worker_id = self.mpi_helper.rank
        self.num_workers = self.mpi_helper.size

        # if agent env is gymnasium-robotics env, should set distance-threshold
        # attr to tolerance
        if hasattr(self.agent.env, "distance_threshold"):
            self.agent.env.__setattr__("distance_threshold", self.tolerance)

        # reset state environment to initiate normalizers
        _,_ = self.agent.env.reset()

        # get goal shape to pass to agent to initialize normalizers
        goal_shape = self.desired_goal_func(self.agent.env).shape
        # instantiate state and goal normalizer objects in agent
        self.agent._init_her(goal_shape, eps=self.normalizer_eps, clip_range=self.normalizer_clip)

        # #sync networks
        # helper.sync_networks(self.agent.actor_model)
        # helper.sync_networks(self.agent.critic_model)


    @classmethod
    def build(
        cls,
        env,
        actor_cnn_layers,
        critic_cnn_layers,
        actor_layers,
        critic_state_layers,
        critic_merged_layers,
        callbacks,
        config,#: wandb.config,
        save_dir: str = "models/",
    ):
        """Builds the agent."""
        # Actor
        actor_learning_rate=config[config.model_type][f"{config.model_type}_actor_learning_rate"]
        actor_optimizer = config[config.model_type][f"{config.model_type}_actor_optimizer"]
        # get optimizer params
        actor_optimizer_params = {}
        if actor_optimizer == "Adam":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
        
        elif actor_optimizer == "Adagrad":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            actor_optimizer_params['lr_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_lr_decay']
        
        elif actor_optimizer == "RMSprop" or actor_optimizer == "SGD":
            actor_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_weight_decay']
            actor_optimizer_params['momentum'] = \
                config[config.model_type][f"{config.model_type}_actor_optimizer_{actor_optimizer}_options"][f'{actor_optimizer}_momentum']

        actor_normalize_layers = config[config.model_type][f"{config.model_type}_actor_normalize_layers"]

        # Critic
        critic_learning_rate=config[config.model_type][f"{config.model_type}_critic_learning_rate"]
        critic_optimizer = config[config.model_type][f"{config.model_type}_critic_optimizer"]
        critic_optimizer_params = {}
        if critic_optimizer == "Adam":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
        
        elif critic_optimizer == "Adagrad":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            critic_optimizer_params['lr_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_lr_decay']
        
        elif critic_optimizer == "RMSprop" or critic_optimizer == "SGD":
            critic_optimizer_params['weight_decay'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_weight_decay']
            critic_optimizer_params['momentum'] = \
                config[config.model_type][f"{config.model_type}_critic_optimizer_{critic_optimizer}_options"][f'{critic_optimizer}_momentum']
        
        critic_normalize_layers = config[config.model_type][f"{config.model_type}_critic_normalize_layers"]

        # Check if CNN layers and if so, build CNN model
        if actor_cnn_layers:
            actor_cnn_model = cnn_models.CNN(actor_cnn_layers, env)
        else:
            actor_cnn_model = None

        if critic_cnn_layers:
            critic_cnn_model = cnn_models.CNN(critic_cnn_layers, env)
        else:
            critic_cnn_model = None
        
        actor_model = models.ActorModel(env = env,
                                        cnn_model = actor_cnn_model,
                                        dense_layers = actor_layers,
                                        optimizer = actor_optimizer,
                                        optimizer_params = actor_optimizer_params,
                                        learning_rate = actor_learning_rate,
                                        normalize_layers = actor_normalize_layers
        )
        critic_model = models.CriticModel(env = env,
                                          cnn_model = critic_cnn_model,
                                          state_layers = critic_state_layers,
                                          merged_layers = critic_merged_layers,
                                          optimizer = critic_optimizer,
                                          optimizer_params = critic_optimizer_params,
                                          learning_rate = critic_learning_rate,
                                          normalize_layers = critic_normalize_layers
        )

        # get goal metrics
        strategy = config[config.model_type][f"{config.model_type}_goal_strategy"]
        tolerance = config[config.model_type][f"{config.model_type}_goal_tolerance"]
        num_goals = config[config.model_type][f"{config.model_type}_num_goals"]

        # get normalizer clip value
        normalizer_clip = config[config.model_type][f"{config.model_type}_normalizer_clip"]

        # get action epsilon
        action_epsilon = config[config.model_type][f"{config.model_type}_epsilon_greedy"]

        # get desired, achieved, reward func for env
        desired_goal_func, achieved_goal_func, reward_func = gym_helper.get_her_goal_functions(env)
        goal_shape = desired_goal_func(env).shape

        ddpg_agent= DDPG(
            env = env,
            actor_model = actor_model,
            critic_model = critic_model,
            discount = config[config.model_type][f"{config.model_type}_discount"],
            tau = config[config.model_type][f"{config.model_type}_tau"],
            action_epsilon = action_epsilon,
            replay_buffer = helper.ReplayBuffer(env=env, buffer_size=100000, goal_shape=goal_shape),
            batch_size = config[config.model_type][f"{config.model_type}_batch_size"],
            noise = helper.Noise.create_instance(config[config.model_type][f"{config.model_type}_noise"], shape=env.action_space.shape, **config[config.model_type][f"{config.model_type}_noise_{config[config.model_type][f'{config.model_type}_noise']}"]),
            callbacks = callbacks,
            save_dir = save_dir,
        )

        return cls(
            agent = ddpg_agent,
            strategy = strategy,
            tolerance = tolerance,
            num_goals = num_goals,
            desired_goal = desired_goal_func,
            achieved_goal = achieved_goal_func,
            reward_fn = reward_func,
            normalizer_clip = normalizer_clip,
        )
        
    def train(self, epochs:int=10, num_cycles:int=50, num_episodes:int=16, num_updates:int=40, render:bool=False, render_freq:int=10, save_dir: str = None):
        

        # set models to train mode
        self.agent.actor_model.train()
        self.agent.critic_model.train()

        if save_dir:
            self.save_dir = save_dir
        if self.agent.callbacks:
            if MPI.COMM_WORLD.Get_rank() == 0:
                for callback in self.agent.callbacks:
                    if isinstance(callback, rl_callbacks.WandbCallback):
                        callback.on_train_begin((self.agent.critic_model, self.agent.actor_model,), logs=self.agent._config)

                    else:
                        callback.on_train_begin(logs=self.agent._config)

        # instantiate new environment
        self.agent.env = self.agent._initialize_env(render, render_freq, context='train')
        
        # initialize step counter (for logging)
        self._step_counter = 0
        self._episode_counter = 0
        self._cycle_counter = 0
        self._success_counter = 0.0
        # set best reward
        # best_reward = self.agent.env.reward_range[0] # substitute with -np.inf
        best_reward = -np.inf
        # instantiate list to store reward history
        reward_history = []
        # instantiate lists to store time history
        episode_time_history = []
        step_time_history = []
        learning_time_history = []
        steps_per_episode_history = []  # List to store steps per episode
        for epoch in range(epochs):
            for cycle in range(num_cycles):
                self._cycle_counter += 1
                for episode in range(num_episodes):
                    # print(f'episode: {episode}')
                    self._episode_counter += 1
                    if self.agent.callbacks:
                        if MPI.COMM_WORLD.Get_rank() == 0:
                            for callback in self.agent.callbacks:
                                callback.on_train_epoch_begin(epoch=self._step_counter, logs=None)
                    episode_start_time = time.time()
                    
                    # reset noise
                    if type(self.agent.noise) == helper.OUNoise:
                        self.agent.noise.reset()
                    


                    # RUN_EPISODE()
                    # reset environment
                    state, _ = self.agent.env.reset()
                    # print(f'state: {state}' )
                    if isinstance(state, dict): # if state is a dict, extract observation (robotics)
                        state = state['observation']
                        # print(f'state extracted obs: {state}')
                    
                    # instantiate empty lists to store current episode trajectory
                    states, actions, next_states, dones, state_achieved_goals, \
                    next_state_achieved_goals, desired_goals = [], [], [], [], [], [], []
                    
                    # set desired goal
                    desired_goal = self.desired_goal_func(self.agent.env)
                    # print(f'desired goal: {desired_goal}')
                    
                    # set achieved goal
                    state_achieved_goal = self.achieved_goal_func(self.agent.env)
                    # print(f'achieved goal: {state_achieved_goal}')
                    
                    # add initial state and goals to local normalizer stats
                    self.agent.state_normalizer.update_local_stats(state)
                    self.agent.goal_normalizer.update_local_stats(desired_goal)
                    self.agent.goal_normalizer.update_local_stats(state_achieved_goal)
                    
                    # set done flag
                    done = False
                    
                    # reset episode reward to 0
                    episode_reward = 0
                    
                    # reset steps counter for the episode
                    episode_steps = 0

                    while not done:
                        # increase step counter
                        self._step_counter += 1
                        
                        # start step timer
                        step_start_time = time.time()
                        
                        # get action
                        action = self.agent.get_action(state, desired_goal, grad=True)
                        
                        # take action
                        next_state, reward, term, trunc, _ = self.agent.env.step(action)
                        # print(f'next state: {next_state}')
                        
                        # extract observation from next state if next_state is dict (robotics)
                        if isinstance(next_state, dict):
                            next_state = next_state['observation']
                            # print(f'new next state: {next_state}')
                        
                        # calculate and log step time
                        step_time = time.time() - step_start_time
                        step_time_history.append(step_time)
                        
                        # get next state achieved goal
                        next_state_achieved_goal = self.achieved_goal_func(self.agent.env)
                        # print(f'next state achieved goal: {next_state_achieved_goal}')
                        
                        # add next state and next state achieved goal to normalizers
                        self.agent.state_normalizer.update_local_stats(next_state)
                        self.agent.goal_normalizer.update_local_stats(next_state_achieved_goal)
                        
                        # calculate distance from achieved goal to desired goal
                        distance_to_goal = np.linalg.norm(
                            self.desired_goal_func(self.agent.env) - self.achieved_goal_func(self.agent.env)
                        )
                        
                        # store distance in step config to send to wandb
                        self.agent._train_step_config["goal distance"] = distance_to_goal
                        
                        # store trajectory in replay buffer (non normalized!)
                        self.agent.replay_buffer.add(state, action, reward, next_state, done,\
                                                        state_achieved_goal, next_state_achieved_goal, desired_goal)
                        
                        # append step state, action, next state, and goals to respective lists
                        states.append(state)
                        actions.append(action)
                        next_states.append(next_state)
                        dones.append(done)
                        state_achieved_goals.append(state_achieved_goal)
                        next_state_achieved_goals.append(next_state_achieved_goal)
                        desired_goals.append(desired_goal)

                        # add to episode reward and increment steps counter
                        episode_reward += reward
                        episode_steps += 1
                        # update state and state achieved goal
                        state = next_state
                        state_achieved_goal = next_state_achieved_goal
                        # update done flag
                        if term or trunc:
                            done = True
                        # log step metrics
                        self.agent._train_step_config["step reward"] = reward
                        self.agent._train_step_config["step time"] = step_time
                        
                        # log to wandb if using wandb callback
                        if self.agent.callbacks:
                            # average step logs across all agents
                            averaged_metrics = helper.sync_metrics(self.agent._train_step_config)
                            # only have the main process log callback values to avoid multiple callback calls
                            if MPI.COMM_WORLD.Get_rank() == 0:
                                for callback in self.agent.callbacks:
                                    callback.on_train_step_end(step=self._step_counter, logs=averaged_metrics)
                        if not done:
                            self._step_counter += 1

                    # calculate success rate
                    # goal_distance = np.linalg.norm(next_state_achieved_goal - desired_goal, axis=-1)
                    success = (distance_to_goal <= self.tolerance).astype(np.float32)
                    self._success_counter += success
                    success_perc = self._success_counter / self._episode_counter
                    # store success rate to train episode config
                    self.agent._train_episode_config["success rate"] = success_perc

                    # update global normalizer stats
                    self.agent.state_normalizer.update_global_stats()
                    self.agent.goal_normalizer.update_global_stats()
                    
                    # package episode states, actions, next states, and goals into trajectory tuple
                    trajectory = (states, actions, next_states, dones, state_achieved_goals, next_state_achieved_goals, desired_goals)

                    # store hindsight experience replay trajectory using current episode trajectory and goal strategy
                    self.store_hindsight_trajectory(trajectory)

                        
                    # check if enough samples in replay buffer and if so, learn from experiences
                    if self.agent.replay_buffer.counter > self.agent.batch_size:
                        learn_time = time.time()
                        for _ in range(num_updates):
                            actor_loss, critic_loss = self.agent.learn()
                        self.agent._train_episode_config["actor loss"] = actor_loss
                        self.agent._train_episode_config["critic loss"] = critic_loss
                
                        learning_time_history.append(time.time() - learn_time)
                    
                    episode_time = time.time() - episode_start_time
                    episode_time_history.append(episode_time)
                    reward_history.append(episode_reward)
                    steps_per_episode_history.append(episode_steps) 
                    avg_reward = np.mean(reward_history[-100:])
                    avg_episode_time = np.mean(episode_time_history[-100:])
                    avg_step_time = np.mean(step_time_history[-100:])
                    avg_learn_time = np.mean(learning_time_history[-100:])
                    avg_steps_per_episode = np.mean(steps_per_episode_history[-100:])  # Calculate average steps per episode

                    self.agent._train_episode_config['episode'] = episode
                    self.agent._train_episode_config["episode reward"] = episode_reward
                    self.agent._train_episode_config["avg reward"] = avg_reward
                    self.agent._train_episode_config["episode time"] = episode_time

                    # log to wandb if using wandb callback
                    if self.agent.callbacks:
                        # average episode logs across all agents
                        averaged_metrics = helper.sync_metrics(self.agent._train_episode_config)
                    
                    # check if best reward
                    if MPI.COMM_WORLD.Get_rank() == 0: # only use main process
                        if averaged_metrics['avg reward'] > best_reward:
                            best_reward = averaged_metrics['avg reward']
                            averaged_metrics["best"] = True
                            # save model
                            self.save()
                        else:
                            averaged_metrics["best"] = False

                        if self.agent.callbacks:
                            for callback in self.agent.callbacks:
                                callback.on_train_epoch_end(epoch=self._step_counter, logs=averaged_metrics)


            # perform soft update on target networks
            self.agent.soft_update(self.agent.actor_model, self.agent.target_actor_model)
            self.agent.soft_update(self.agent.critic_model, self.agent.target_critic_model)

            # print metrics to terminal log
            if MPI.COMM_WORLD.Get_rank() == 0: # only use main process
                print(f"epoch {epoch} cycle {self._cycle_counter} episode {self._episode_counter}, reward {averaged_metrics['episode reward']}, avg reward {averaged_metrics['avg reward']}, avg episode time {avg_episode_time:.2f}s")

        # if callbacks, call on train end
        if MPI.COMM_WORLD.Get_rank() == 0: # only use main process
            if self.agent.callbacks:
                for callback in self.agent.callbacks:
                    callback.on_train_end(logs=averaged_metrics)
        # close the environment
        self.agent.env.close()

    def test(self, num_episodes, render, render_freq):
        """Runs a test over 'num_episodes'."""

        # set model in eval mode
        self.agent.actor_model.eval()
        self.agent.critic_model.eval()

        if self.agent.callbacks:
            for callback in self.agent.callbacks:
                callback.on_test_begin(logs=self.agent._config)

        # instantiate new environment
        self.agent.env = self.agent._initialize_env(render, render_freq, context='test')

        # instantiate list to store reward, step time, and episode time history
        reward_history = []
        step_time_history = []
        self._step = 1
        # set the model to calculate no gradients during evaluation
        with torch.no_grad():
            for i in range(num_episodes):
                if self.agent.callbacks:
                    for callback in self.agent.callbacks:
                        callback.on_test_epoch_begin(epoch=self._step, logs=None) # update to pass any logs if needed

                state, _ = self.agent.env.reset()
                # set desired goal
                desired_goal = self.desired_goal_func(self.agent.env)
                done = False
                episode_reward = 0
                while not done:
                    # start step timer
                    step_start_time = time.time()
                    # get normalized values for state and desired goal
                    state = self.agent.state_normalizer.normalize(state)
                    desired_goal = self.agent.goal_normalizer.normalize(desired_goal)
                    # get action
                    action = self.agent.get_action(state, desired_goal, grad=False, test=True)
                    next_state, reward, term, trunc, _ = self.agent.env.step(action)
                    # extract observation from next state if next_state is dict (robotics)
                    if isinstance(next_state, dict):
                        next_state = next_state['observation']
                    # calculate and log step time
                    step_time = time.time() - step_start_time
                    step_time_history.append(step_time)
                    if term or trunc:
                        done = True
                    episode_reward += reward
                    state = next_state
                    self._step += 1

                reward_history.append(episode_reward)
                avg_reward = np.mean(reward_history[-100:])
                self.agent._test_episode_config["episode_reward"] = episode_reward
                self.agent._test_episode_config["avg_reward"] = avg_reward
                if self.agent.callbacks:
                    for callback in self.agent.callbacks:
                        callback.on_test_epoch_end(epoch=self._step, logs=self.agent._test_episode_config)

                print(f"episode {i+1}, score {episode_reward}, avg_score {avg_reward}")

            if self.agent.callbacks:
                for callback in self.agent.callbacks:
                    callback.on_test_end(logs=self.agent._test_episode_config)
            # close the environment
            self.agent.env.close()

    def store_hindsight_trajectory(self, trajectory):
        """
        Stores a hindsight experience replay trajectory in the replay buffer.
        Args:
            trajectory: tuple of states, actions, next states, dones, state achieved goals, next state achieved goals, and desired goals
        """
        # with self.lock:
        states, actions, next_states, dones, state_achieved_goals, next_state_achieved_goals, desired_goals = trajectory
        # print(f'states: {states}')
        # print(f'actions: {actions}')
        # print(f'next states: {next_states}')
        # print(f'dones: {dones}')
        # print(f'state achieved goals: {state_achieved_goals}')
        # print(f'next state achieved goals: {next_state_achieved_goals}')
        # print(f'desired goals: {desired_goals}')
        # instantiate variable to keep track of times tolerance is hit
        tol_count = 0

        # loop over each step in the trajectory to set new achieved goals, calculate new reward, and save to replay buffer
        for idx, (state, action, next_state, done, state_achieved_goal, next_state_achieved_goal, desired_goal) in enumerate(zip(states, actions, next_states, dones, state_achieved_goals, next_state_achieved_goals, desired_goals)):
            
            # normalize state and next states goals
            # state_achieved_goal_norm = self.agent.goal_normalizer.normalize(state_achieved_goal)
            # next_state_achieved_goal_norm = self.agent.goal_normalizer.normalize(next_state_achieved_goal)

            

            if self.strategy == "final":
                new_desired_goal = next_state_achieved_goals[-1]
                # normalize desired goal to pass to reward func
                # new_desired_goal_norm = self.agent.goal_normalizer.normalize(new_desired_goal)
                new_reward, within_tol = self.reward_fn(self.agent.env, action, state_achieved_goal, next_state_achieved_goal, new_desired_goal, self.tolerance)
                # increment tol_count
                tol_count += within_tol

                # store non normalized trajectory
                self.agent.replay_buffer.add(state, action, new_reward, next_state, done, state_achieved_goal, next_state_achieved_goal, new_desired_goal)

            elif self.strategy == 'future':
                for i in range(self.num_goals):
                    if idx + i >= len(states) -1:
                        break
                    goal_idx = np.random.randint(idx + 1, len(states))
                    new_desired_goal = next_state_achieved_goals[goal_idx]
                    # normalize desired goal to pass to reward func
                    # new_desired_goal_norm = self.agent.goal_normalizer.normalize(new_desired_goal)
                    new_reward, within_tol = self.reward_fn(self.agent.env, action, state_achieved_goal, next_state_achieved_goal, new_desired_goal, self.tolerance)
                    # print(f'new reward: {new_reward}')
                    # print(f'within tol: {within_tol}')
                    # increment tol_count
                    tol_count += within_tol
                    # print(f'tol count: {tol_count}')
                    # store non normalized trajectory
                    self.agent.replay_buffer.add(state, action, new_reward, next_state, done, state_achieved_goal, next_state_achieved_goal, new_desired_goal)

            elif self.strategy == 'none':
                break

        # add tol count to train step config for callbacks
        if self.agent.callbacks:
            self.agent._train_episode_config["tolerance count"] = tol_count
                
        

    def set_normalizer_state(self, config):
        self.agent.state_normalizer.set_state(config)        


    def get_config(self):
        config = {
            "agent_type": self.__class__.__name__,
            "agent": self.agent.get_config(),
            "strategy": self.strategy,
            "tolerance":self.tolerance,
            "num_goals": self.num_goals,
            "desired_goal": self.desired_goal_func.__name__,
            "achieved_goal": self.achieved_goal_func.__name__,
            "reward_fn": self.reward_fn.__name__,
            "normalizer_clip": self.normalizer_clip,
            "normalizer_eps": self.normalizer_eps,
            "save_dir": self.save_dir,
        }

        # if callable(self.reward_fn) and self.reward_fn.__name__ == '<lambda>':
        #     config["reward_fn"] = inspect.getsource(self.reward_fn).strip()
        # else:
        #     config["reward_fn"] = self.reward_fn.__name__

        return config
    
    def save(self):
        """Saves the model."""
        obj_config = self.get_config()

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of DDPG agent config
        with open(self.save_dir + "/obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

        # save agent
        self.agent.save()

        self.agent.state_normalizer.save_state(self.save_dir + "/state_normalizer.npz")
        self.agent.goal_normalizer.save_state(self.save_dir + "/goal_normalizer.npz")

    @classmethod
    def load(cls, folder: str = "models/her", load_weights=True):
        """Loads the model."""
        # load reinforce agent config
        with open(
            Path(folder).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
        ) as f:
            obj_config = json.load(f)

        # Resolve function names to actual functions
        obj_config["desired_goal"] = getattr(gym_helper, obj_config["desired_goal"])
        obj_config["achieved_goal"] = getattr(gym_helper, obj_config["achieved_goal"])
        obj_config["reward_fn"] = getattr(gym_helper, obj_config["reward_fn"])


        # load agent
        agent = load_agent_from_config(obj_config["agent"]["save_dir"], load_weights)

        # instantiate HER model
        her = cls(agent, obj_config["strategy"], obj_config["tolerance"], obj_config["num_goals"],
                  obj_config["desired_goal"], obj_config["achieved_goal"], obj_config["reward_fn"],
                  obj_config['normalizer_clip'], obj_config['normalizer_eps'], obj_config["save_dir"])

        # load agent normalizers
        agent.state_normalizer = helper.Normalizer.load_state(folder + "/state_normalizer.npz")
        agent.goal_normalizer = helper.Normalizer.load_state(folder + "/goal_normalizer.npz")
        
        return her
    

def load_agent_from_config(config_path, load_weights=True):
    """Loads an agent from a config file."""
    with open(
        Path(config_path).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
    ) as f:
        config = json.load(f)

    agent_type = config["agent_type"]

    # Use globals() to get a reference to the class
    agent_class = globals().get(agent_type)

    if agent_class:
        return agent_class.load(config_path, load_weights)

    raise ValueError(f"Unknown agent type: {agent_type}")


def get_agent_class_from_type(agent_type: str):
    """Builds an agent from a passed agent type str."""

    types = {"Actor Critic": "ActorCritic",
             "Reinforce": "Reinforce",
             "DDPG": "DDPG",
             "HER_DDPG": "HER",
            }

    # Use globals() to get a reference to the class
    agent_class = globals().get(types[agent_type])

    if agent_class:
        return agent_class

    raise ValueError(f"Unknown agent type: {agent_type}")
