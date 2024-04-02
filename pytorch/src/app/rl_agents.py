"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
import json
import os
from typing import List
from pathlib import Path
import time
import datetime

import rl_callbacks
import models
import wandb
import wandb_support
import helper
import dash_callbacks

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
        replay_buffer: helper.ReplayBuffer = None,
        batch_size: int = 64,
        noise = None,
        callbacks: List = [],
        save_dir: str = "models/",
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
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.noise = noise
        
        self.save_dir = save_dir
        self._DEBUG = _DEBUG
        
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

    def clone_model(self, model):
        """Clones a model."""
        return model.get_clone()
    
    @classmethod
    def build(
        cls,
        env,
        actor_layers,
        critic_state_layers,
        critic_merged_layers,
        callbacks,
        config,#: wandb.config,
        save_dir: str = "models/",
    ):
        """Builds the agent."""
        actor_optimizer = helper.get_optimizer_by_name(config[config.model_type][f"{config.model_type}_actor_optimizer"]) 
        critic_optimizer = helper.get_optimizer_by_name(config[config.model_type][f"{config.model_type}_critic_optimizer"])
        actor_model = models.ActorModel(
            env=env, dense_layers=actor_layers, learning_rate=config[config.model_type][f"{config.model_type}_actor_learning_rate"], optimizer=actor_optimizer
        )
        critic_model = models.CriticModel(
            env=env, state_layers=critic_state_layers, merged_layers=critic_merged_layers, learning_rate=config[config.model_type][f"{config.model_type}_critic_learning_rate"], optimizer=critic_optimizer
        )

        return cls(
            env = env,
            actor_model = actor_model,
            critic_model = critic_model,
            discount = config[config.model_type][f"{config.model_type}_discount"],
            tau = config[config.model_type][f"{config.model_type}_tau"],
            replay_buffer = helper.Buffer.create_instance(config[config.model_type][f"{config.model_type}_replay_buffer"], env=env),
            batch_size = config[config.model_type][f"{config.model_type}_batch_size"],
            noise = helper.Noise.create_instance(config[config.model_type][f"{config.model_type}_noise"], shape=env.action_space.shape, **config[config.model_type][f"{config.model_type}_noise_{config[config.model_type][f'{config.model_type}_noise']}"]),
            callbacks = callbacks,
            save_dir = save_dir,
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
    
    def get_action(self, state):
        # make sure state is a tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.actor_model.device)

        # permute state to (C,H,W) if actor using cnn model
        if self.actor_model.cnn_model:
            state = state.permute(2, 0, 1).unsqueeze(0)
            # print(f'permuted state shape: {state.size()}')

        action_value = self.actor_model(state)
        noise = self.noise(action_value.size())

        # Convert the action space bounds to a tensor on the same device
        action_space_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.actor_model.device)
        action_space_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.actor_model.device)

        action = (action_value + noise).clip(action_space_low, action_space_high)

        noise_np = noise.cpu().detach().numpy().flatten()
        action_np = action.cpu().detach().numpy().flatten()

        # Loop over the noise and action values and log them to wandb
        for i, (a,n) in enumerate(zip(action_np, noise_np)):
            # Log the values to wandb
            self._train_step_config[f'action_{i}'] = a
            self._train_step_config[f'noise_{i}'] = n

        return action_np


    def learn(self):
        # run callbacks on train batch begin
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_train_step_begin(step=self._step, logs=None)
        
        with torch.no_grad():
            # sample a batch of experiences from the replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # permute states and next states if using cnn
            if self.actor_model.cnn_model:
                states = states.permute(0, 3, 1, 2)
                next_states = next_states.permute(0, 3, 1, 2)

            # convert rewards and dones to 2d tensors
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)

            # get target values 
            target_actions = self.target_actor_model(next_states)
            target_critic_values = self.target_critic_model(next_states, target_actions)
            targets = rewards + self.discount * target_critic_values * (1 - dones)
        
        # get current critic values and calculate critic loss
        prediction = self.critic_model(states, actions)
        critic_loss = F.mse_loss(prediction, targets)
        
        # update critic
        self.critic_model.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_model.optimizer.step()
        
        # update actor
        action_values = self.actor_model(states)
        critic_values = self.critic_model(states, action_values)
        actor_loss = -critic_values.mean()
        self.actor_model.optimizer.zero_grad()
        actor_loss.backward()
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
                self.noise.reset(torch.tensor(self.env.action_space.sample()).size())
            # reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0  # Initialize steps counter for the episode
            while not done:
                step_start_time = time.time()
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action) # might have to use action.numpy()
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
                    action = self.get_action(state)
                    next_state, reward, term, trunc, _ = self.env.step(action.numpy(force=True))
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
                "replay_buffer": self.replay_buffer.get_config(),
                "batch_size": self.batch_size,
                "noise": self.noise.get_config(),
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

        # if wandb callback, save wandb config
        # if self._wandb:
        #     for callback in self.callbacks:
        #         if isinstance(callback, rl_callbacks.WandbCallback):
        #             callback.save(self.save_dir + "/wandb_config.json")

    @classmethod
    def load(cls, folder: str = "models", load_weights=True):
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
        # load callbacks
        callbacks = [rl_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in obj_config['callbacks']]

        # return DDPG agent
        agent = cls(
            gym.make(obj_config["env"]),
            actor_model = actor_model,
            critic_model = critic_model,
            discount=obj_config["discount"],
            tau=obj_config["tau"],
            replay_buffer=replay_buffer, # need to change to load from base class
            batch_size=obj_config["batch_size"],
            noise=noise, # need to change to load from base class
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )
        
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, callbacks.WandbCallback):
                    agent._wandb = True
                    agent._train_config = {}
                    agent._train_episode_config = {}
                    agent._train_step_config = {}
                    agent._test_config = {}

        return agent


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
             "DDPG": "DDPG"}

    # Use globals() to get a reference to the class
    agent_class = globals().get(types[agent_type])

    if agent_class:
        return agent_class

    raise ValueError(f"Unknown agent type: {agent_type}")
