"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
import json
import os
from typing import List
from pathlib import Path
import time
import datetime

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import CallbackList, Callback
from tensorflow.keras.models import clone_model

import models
import wandb
import wandb_support
import helper
import dash_callbacks
import tf_callbacks


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
        learning_rate: float = 1e-5,
        discount=0.99,
        policy_trace_decay: float = 0.0,
        value_trace_decay: float = 0.0,
        callbacks: List[Callback] = None,
        save_dir: str = "models/",
    ):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.discount = discount
        self.policy_trace_decay = policy_trace_decay
        self.value_trace_decay = value_trace_decay
        self.callbacks = callbacks
        self.save_dir = save_dir
        if callbacks:
            self.callback_list = self._create_callback_list(callbacks)
            for callback in callbacks:
                if isinstance(callback, tf_callbacks.WandbCallback):
                    self._config = callback._config(self)
                    self._wandb = True
                    self._train_config = {}
                    self._train_episode_config = {}
                    self._train_step_config = {}
                    self._test_config = {}
                    self._test_episode_config = {}
                    break

                self._wandb = False
                self._config = {}
        else:
            self.callback_list = None
        # set learning rate for optimizers
        self.policy_model._set_learning_rate(self.learning_rate)
        self.value_model._set_learning_rate(self.learning_rate)
        # set self.action to None
        self.action = None
        # instantiate and set policy and value traces
        self.policy_trace = []
        self.value_trace = []
        self._set_traces()
        # instantiant influence variable for policy trace updates
        self.influence = 1
        # instantiate and set keras loss objects
        self.policy_loss = tf.keras.metrics.Mean(name="Policy Loss")
        self.value_loss = tf.keras.metrics.Mean(name="Value Loss")

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
        policy_optimizer = helper.get_optimizer_by_name(wandb.config.policy_optimizer)
        value_optimizer = helper.get_optimizer_by_name(wandb.config.value_optimizer)
        policy_model = models.PolicyModel(
            env, dense_layers=policy_layers, optimizer=policy_optimizer
        )
        value_model = models.ValueModel(
            env, hidden_layers=value_layers, optimizer=value_optimizer
        )

        return cls(
            env,
            policy_model,
            value_model,
            config.learning_rate,
            config.discount,
            config.policy_trace_decay,
            config.value_trace_decay,
            callbacks,
            save_dir=save_dir,
        )

    def _initialize_env(self, render=False, render_freq=10):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                env,
                self.save_dir + "/renders",
                episode_trigger=lambda episode_id: episode_id % render_freq == 0,
            )

        return gym.make(self.env.spec)

    def _set_traces(self):
        for weights in self.policy_model.trainable_variables:
            self.policy_trace.append(np.zeros_like(weights))
        for weights in self.value_model.trainable_variables:
            self.value_trace.append(np.zeros_like(weights))

    def _update_value_trace(self, value_gradient):
        for i, _ in enumerate(self.value_trace):
            self.value_trace[i] = (
                self.discount * self.value_trace_decay * self.value_trace[i]
                + value_gradient[i]
            )

    def _update_policy_trace(self, policy_gradient):
        for i, _ in enumerate(self.policy_trace):
            self.policy_trace[i] = (
                self.discount * self.policy_trace_decay * self.policy_trace[i]
            ) + (self.influence * policy_gradient[i])

    def _create_callback_list(self, callbacks):
        if callbacks is None:
            callbacks = []
        callback_list = CallbackList(callbacks)
        return callback_list

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probabilities = self.policy_model(state)
        probabilities_dist = tfp.distributions.Categorical(probs=probabilities)
        action = probabilities_dist.sample()
        self.action = action

        return action.numpy()[0]

    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None
    ):
        """Trains the model for 'episodes' number of episodes."""
        # set save_dir if not None
        if save_dir:
            self.save_dir = save_dir
        if self.callback_list:
            self.callback_list.on_train_begin(logs=self._config)
        reward_history = []
        self.env = self._initialize_env(render, render_freq)
        # set step counter
        step = 1
        # set best reward
        best_reward = self.env.reward_range[0]

        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_epoch_begin(epoch=step, logs=None)
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            self.influence = 1
            while not done:
                if self._wandb:
                    self.callback_list.on_train_batch_begin(batch=step, logs=None)
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                if term or trunc:
                    done = True
                episode_reward += reward
                self.learn(state, reward, next_state, done)
                if self._wandb:
                    self._train_step_config["action"] = action
                    self._train_step_config["step_reward"] = reward
                    self._train_step_config["influence"] = self.influence
                    # each state value
                    for e, s in enumerate(state):
                        self._train_step_config[f"state_{e}"] = s
                if self.callback_list:
                    self.callback_list.on_train_batch_end(
                        batch=step, logs=self._train_step_config
                    )
                state = next_state
                self.influence *= self.discount
                step += 1
            reward_history.append(episode_reward)
            self._train_episode_config["episode_reward"] = episode_reward
            avg_reward = np.mean(reward_history[-100:])
            self._train_episode_config["avg_reward"] = avg_reward
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callback_list:
                self.callback_list.on_epoch_end(
                    epoch=step, logs=self._train_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}")

        # close the environment
        self.env.close()
        self.callback_list.on_train_end(logs=self._train_episode_config)

    def learn(self, state, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as value_tape:
            state_value = self.value_model(state)
            next_state_value = self.value_model(next_state)
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)
            temporal_difference = (
                reward
                + self.discount * next_state_value * (1 - int(done))
                - state_value
            )
            value_loss = temporal_difference**2

        with tf.GradientTape() as policy_tape:
            probabilities = self.policy_model(state)
            probabilities_dist = tfp.distributions.Categorical(probs=probabilities)
            log_probability = (
                probabilities_dist.log_prob(self.action)
                + np.finfo(np.float32).eps.item()
            )

            policy_loss = -log_probability * temporal_difference

        # calculate gradients
        value_gradient = value_tape.gradient(
            value_loss, self.value_model.trainable_variables
        )
        policy_gradient = policy_tape.gradient(
            policy_loss, self.policy_model.trainable_variables
        )
        # update traces
        self._update_value_trace(value_gradient)
        self._update_policy_trace(policy_gradient)
        self.value_model.optimizer.apply_gradients(
            zip(self.value_trace, self.value_model.trainable_variables)
        )
        self.policy_model.optimizer.apply_gradients(
            zip(self.policy_trace, self.policy_model.trainable_variables)
        )
        # append losses to loss metrics
        self.policy_loss(policy_loss)
        self.value_loss(value_loss)
        if self._wandb:
            self._train_step_config["temporal_difference"] = temporal_difference
            for i, probability in enumerate(tf.squeeze(probabilities)):
                self._train_step_config[f"probability_action_{i}"] = probability
            self._train_step_config["log_probability"] = log_probability
            self._train_step_config["policy_loss"] = self.policy_loss.result()
            self._train_step_config["value_loss"] = self.value_loss.result()
            for p_grad, p_trace, v_grad, v_trace in zip(
                policy_gradient, self.policy_trace, value_gradient, self.value_trace
            ):
                self._train_step_config["policy_grad_mean"] = tf.reduce_mean(p_grad)
                self._train_step_config["policy_grad_std"] = tf.math.reduce_std(p_grad)
                self._train_step_config["policy_grad_max"] = tf.reduce_max(p_grad)
                self._train_step_config["policy_grad_min"] = tf.reduce_min(p_grad)
                self._train_step_config["policy_grad_norm"] = tf.norm(p_grad)
                self._train_step_config["policy_trace_mean"] = tf.reduce_mean(p_trace)
                self._train_step_config["policy_trace_std"] = tf.math.reduce_std(
                    p_trace
                )
                self._train_step_config["policy_trace_max"] = tf.reduce_max(p_trace)
                self._train_step_config["policy_trace_min"] = tf.reduce_min(p_trace)
                self._train_step_config["policy_trace_norm"] = tf.norm(p_trace)
                self._train_step_config["value_grad_mean"] = tf.reduce_mean(v_grad)
                self._train_step_config["value_grad_std"] = tf.math.reduce_std(v_grad)
                self._train_step_config["value_grad_max"] = tf.reduce_max(v_grad)
                self._train_step_config["value_grad_min"] = tf.reduce_min(v_grad)
                self._train_step_config["value_grad_norm"] = tf.norm(v_grad)
                self._train_step_config["value_trace_mean"] = tf.reduce_mean(v_trace)
                self._train_step_config["value_trace_std"] = tf.math.reduce_std(v_trace)
                self._train_step_config["value_trace_max"] = tf.reduce_max(v_trace)
                self._train_step_config["value_trace_min"] = tf.reduce_min(v_trace)
                self._train_step_config["value_trace_norm"] = tf.norm(v_trace)

    def test(self, num_episodes, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""
        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq)
        if self.callback_list:
            self.callback_list.on_test_begin(logs=self._config)

        step = 1
        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_test_batch_begin(batch=step, logs=None)
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
                step += 1
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])
            self._test_episode_config["episode_reward"] = episode_reward
            self._test_episode_config["avg_reward"] = avg_reward
            if self.callback_list:
                self.callback_list.on_test_batch_end(
                    batch=step, logs=self._test_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}")

        if self.callback_list:
            self.callback_list.on_test_end(logs=self._test_episode_config)
        # close the environment
        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "learning_rate": self.learning_rate,
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

        # if wandb callback, save wandb config
        if self._wandb:
            for callback in self.callback_list:
                if isinstance(callback, tf_callbacks.WandbCallback):
                    callback.save(self.save_dir + "/wandb_config.json")

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
        # load wandb config if exists
        if os.path.exists(Path(folder).joinpath(Path("wandb_config.json"))):
            callbacks = [
                tf_callbacks.WandbCallback.load(
                    Path(folder).joinpath(Path("wandb_config.json"))
                )
            ]
        else:
            callbacks = None

        # return reinforce agent
        agent = cls(
            gym.make(obj_config["env"]),
            policy_model=policy_model,
            value_model=value_model,
            learning_rate=obj_config["learning_rate"],
            discount=obj_config["discount"],
            policy_trace_decay=obj_config["policy_trace_decay"],
            value_trace_decay=obj_config["value_trace_decay"],
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )

        for callback in callbacks:
            if isinstance(callback, tf_callbacks.WandbCallback):
                agent._wandb = True
                agent._train_config = {}
                agent._train_episode_config = {}
                agent._train_step_config = {}
                agent._test_config = {}

        return agent


class Reinforce(Agent):
    def __init__(
        self,
        env: gym.Env,
        policy_model: models.PolicyModel,
        value_model: models.ValueModel = None,
        learning_rate: float = 1e-5,
        discount=0.99,
        callbacks: List[Callback] = None,
        save_dir: str = "models/",
    ):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.discount = discount
        self.callbacks = callbacks
        self.save_dir = save_dir
        # instantiate and set keras loss objects
        self.policy_loss = tf.keras.metrics.Mean(name="Policy Loss")
        self.value_loss = tf.keras.metrics.Mean(name="Value Loss")
        # set learning rate for optimizers
        self.policy_model._set_learning_rate(self.learning_rate)
        self.value_model._set_learning_rate(self.learning_rate)
        if callbacks:
            self.callback_list = self._create_callback_list(callbacks)
            for callback in self.callback_list:
                if isinstance(callback, tf_callbacks.WandbCallback):
                    self._config = callback._config(self)
                    self._wandb = True
                    self._train_config = {}
                    self._train_episode_config = {}
                    self._train_step_config = {}
                    self._test_config = {}
                    self._test_episode_config = {}
                    break

                self._wandb = False
        else:
            self.callback_list = None

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
        policy_optimizer = helper.get_optimizer_by_name(wandb.config.policy_optimizer)
        value_optimizer = helper.get_optimizer_by_name(wandb.config.value_optimizer)
        policy_model = models.PolicyModel(
            env, dense_layers=policy_layers, optimizer=policy_optimizer
        )
        value_model = models.ValueModel(
            env, hidden_layers=value_layers, optimizer=value_optimizer
        )

        return cls(
            env,
            policy_model,
            value_model,
            config.learning_rate,
            config.discount,
            callbacks,
            save_dir=save_dir,
        )

    def _create_callback_list(self, callbacks):
        if callbacks is None:
            callbacks = []
        callback_list = CallbackList(callbacks)

        return callback_list

    def _initialize_env(self, render=False, render_freq=10):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                env,
                self.save_dir + "/renders",
                episode_trigger=lambda episode_id: episode_id % render_freq == 0,
            )

        return gym.make(self.env.spec)

    def get_return(self, rewards):
        """Compute expected returns per timestep."""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of `rewards` and accumulate reward sums
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = tf.squeeze(reward + self.discount * discounted_sum)
            # discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        return returns

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probabilities = self.policy_model(state)
        probabilities_dist = tfp.distributions.Categorical(probs=probabilities)
        action = probabilities_dist.sample()
        # self.action = action

        return action.numpy()[0]

    def learn(self, states, actions, rewards):
        returns = self.get_return(rewards)
        # instantiate variables to keep track of total value and policy losses over the episode
        total_value_loss = 0
        total_policy_loss = 0
        for t, (state, action, _return, reward, step) in enumerate(
            zip(states, actions, returns, rewards, self._cur_learning_steps)
        ):
            if self.callback_list:
                self.callback_list.on_train_batch_begin(batch=step, logs=None)
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            _return = tf.convert_to_tensor(_return, dtype=tf.float32)
            if self.value_model:
                with tf.GradientTape() as value_tape:
                    state_value = self.value_model(state)
                    state_value = tf.squeeze(state_value)
                    # calculate performance
                    advantage = _return - state_value
                    # calculate value loss
                    value_loss = tf.reduce_sum(advantage**2)

                # calculate gradient
                value_gradient = value_tape.gradient(
                    value_loss, self.value_model.trainable_variables
                )
                # update value model weights
                self.value_model.optimizer.apply_gradients(
                    zip(value_gradient, self.value_model.trainable_variables)
                )
                # add value loss to total value loss
                total_value_loss += value_loss

            with tf.GradientTape() as policy_tape:
                probabilities = self.policy_model(state)
                probabilities_dist = tfp.distributions.Categorical(probs=probabilities)
                log_probability = (
                    probabilities_dist.log_prob(action)
                    + np.finfo(np.float32).eps.item()
                )

                policy_loss = -log_probability * advantage * self.discount**t

            # calculate gradient
            policy_gradient = policy_tape.gradient(
                policy_loss, self.policy_model.trainable_variables
            )
            # update policy model weights
            self.policy_model.optimizer.apply_gradients(
                zip(policy_gradient, self.policy_model.trainable_variables)
            )
            # add policy loss to total policy loss
            total_policy_loss += policy_loss

            # log to wandb if using wandb callback
            if self._wandb:
                self._train_step_config["action"] = action
                self._train_step_config["step_reward"] = reward
                self._train_step_config["temporal_difference"] = advantage
                for i, probability in enumerate(tf.squeeze(probabilities)):
                    self._train_step_config[f"probability_action_{i}"] = probability
                self._train_step_config["log_probability"] = log_probability
                self._train_step_config["policy_loss"] = policy_loss
                self._train_step_config["value_loss"] = value_loss
                for p_grad, v_grad in zip(policy_gradient, value_gradient):
                    self._train_step_config["policy_grad_mean"] = tf.reduce_mean(p_grad)
                    self._train_step_config["policy_grad_std"] = tf.math.reduce_std(
                        p_grad
                    )
                    self._train_step_config["policy_grad_max"] = tf.reduce_max(p_grad)
                    self._train_step_config["policy_grad_min"] = tf.reduce_min(p_grad)
                    self._train_step_config["policy_grad_norm"] = tf.norm(p_grad)
                    self._train_step_config["value_grad_mean"] = tf.reduce_mean(v_grad)
                    self._train_step_config["value_grad_std"] = tf.math.reduce_std(
                        v_grad
                    )
                    self._train_step_config["value_grad_max"] = tf.reduce_max(v_grad)
                    self._train_step_config["value_grad_min"] = tf.reduce_min(v_grad)
                    self._train_step_config["value_grad_norm"] = tf.norm(v_grad)
                    # each state value
                    for e, s in enumerate(state):
                        self._train_step_config[f"state_{e}"] = s

            if self.callback_list:
                self.callback_list.on_train_batch_end(
                    batch=step, logs=self._train_step_config
                )

        # append losses to loss metrics
        self.policy_loss(total_policy_loss)
        self.value_loss(total_value_loss)

    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None
    ):
        """Trains the model for 'episodes' number of episodes."""
        if save_dir:
            self.save_dir = save_dir
        if self.callback_list:
            self.callback_list.on_train_begin(logs=self._config)
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq)
        if self._wandb:
            # set step counter
            self._step = 1
            # set current learning steps
            self._cur_learning_steps = []
        # set best reward
        best_reward = self.env.reward_range[0]
        # instantiate list to store reward history
        reward_history = []
        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_epoch_begin(epoch=self._step, logs=None)
            states = []
            next_states = []
            actions = []
            rewards = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                if self._wandb:
                    self._cur_learning_steps.append(self._step)
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
                if self._wandb and not done:
                    self._step += 1
            self.learn(states, actions, rewards)
            # clear learning steps array
            if self._wandb:
                self._cur_learning_steps = []
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])

            self._train_episode_config["episode_reward"] = episode_reward
            self._train_episode_config["avg_reward"] = avg_reward
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callback_list:
                self.callback_list.on_epoch_end(
                    epoch=self._step, logs=self._train_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}")

        if self.callback_list:
            self.callback_list.on_train_end(logs=self._train_episode_config)
        # close the environment
        self.env.close()

    def test(self, num_episodes, render=False, render_freq=10):
        """Runs a test over 'num_episodes'."""
        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq)
        if self.callback_list:
            self.callback_list.on_test_begin(logs=None)
        self._step = 1
        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_test_batch_begin(batch=self._step, logs=None)
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
            self._test_episode_config["episode_reward"] = episode_reward
            self._test_episode_config["avg_reward"] = avg_reward
            if self.callback_list:
                self.callback_list.on_test_batch_end(
                    batch=self._step, logs=self._test_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}")

        if self.callback_list:
            self.callback_list.on_test_end(logs=self._test_episode_config)
        # close the environment
        self.env.close()

    def get_config(self):
        return {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "policy_model": self.policy_model.get_config(),
            "value_model": self.value_model.get_config(),
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "callbacks": [callback.__class__.__name__ for callback in (self.callbacks or []) if callback is not None],
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
        if self._wandb:
            for callback in self.callback_list:
                if isinstance(callback, tf_callbacks.WandbCallback):
                    callback.save(self.save_dir + "/wandb_config.json")

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
        # load wandb config if exists
        if os.path.exists(Path(folder).joinpath(Path("wandb_config.json"))):
            callbacks = [
                tf_callbacks.WandbCallback.load(
                    Path(folder).joinpath(Path("wandb_config.json"))
                )
            ]
        else:
            callbacks = None

        # return reinforce agent
        agent = cls(
            gym.make(obj_config["env"]),
            policy_model=policy_model,
            value_model=value_model,
            learning_rate=obj_config["learning_rate"],
            discount=obj_config["discount"],
            callbacks=callbacks,
            save_dir=obj_config["save_dir"],
        )

        for callback in callbacks:
            if isinstance(callback, tf_callbacks.WandbCallback):
                agent._wandb = True
                agent._train_config = {}
                agent._train_episode_config = {}
                agent._train_step_config = {}
                agent._test_config = {}

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
        callbacks: List[Callback] = None,
        save_dir: str = "models/",
        _DEBUG: bool = False
    ):
        self.env = env
        self.actor_model = actor_model
        self.critic_model = critic_model
        # set target actor and critic models
        self.target_actor_model = self.clone_model(self.actor_model)
        self.target_critic_model = self.clone_model(self.critic_model)
        # self.target_actor_model = clone_model(self.actor_model)
        # self.target_critic_model = clone_model(self.critic_model)
        # # set weights of target models
        # self.target_actor_model.set_weights(self.actor_model.get_weights())
        # self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.discount = discount
        self.tau = tau
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.noise = noise
        
        self.save_dir = save_dir
        self._DEBUG = _DEBUG
        
        self.critic_loss = tf.keras.losses.MeanSquaredError(name="Critic_Loss")
        # self.critic_loss_metric = tf.keras.metrics.Mean(name="Critic_Loss")
        # self.actor_loss = tf.keras.metrics.Mean(name="Actor_Loss")
        # setup callbacks
        self.train_log_dir = None # for tensorboard
        self.test_log_dir = None # for tensorboard
        self.callbacks = callbacks
        # self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='tensorboard/logs/'))
        if callbacks:
            self.callback_list = self._create_callback_list(callbacks)
            for callback in self.callback_list.callbacks:
                if isinstance(callback, tf_callbacks.WandbCallback):
                    self._config = callback._config(self)
                    self._wandb = True
                    break

        else:
            self.callback_list = None
            self._wandb = False
        self._train_config = {}
        self._train_episode_config = {}
        self._train_step_config = {}
        self._test_config = {}
        self._test_episode_config = {}

        self._step = None
        # instantiate and copy weights for target actor and critic models

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

    def _create_callback_list(self, callbacks):
        if callbacks is None:
            callbacks = []
        callback_list = CallbackList(callbacks)

        return callback_list

    def _initialize_env(self, render=False, render_freq=10, context=None):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec, render_mode="rgb_array")
            if context == "train":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/training",
                    episode_trigger=lambda episode_id: episode_id % render_freq == 0,
                )
            elif context == "test":
                return gym.wrappers.RecordVideo(
                    env,
                    self.save_dir + "/renders/testing",
                    episode_trigger=lambda episode_id: episode_id % render_freq == 0,
                )

        return gym.make(self.env.spec)
    
    def get_action(self, state):
        # receives current state and returns a vector of action values from policy model
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_value = self.actor_model(state)[0]
        noise = self.noise()
        action = tf.clip_by_value((action_value + noise), self.env.action_space.low, self.env.action_space.high)
        for i, (n,a) in enumerate(zip(noise, action)):
            self._train_step_config[f'noise {i}'] = n
            self._train_step_config[f'action {i}'] = a
        
        # DEBUG
        # print(f'step: {self._step}')
        # print(f'action_value: {action_value}')
        # print(f'noise: {noise}')
        # print(f'action_value + noise: {action_value + noise}')
        # print(f'clipped action: {action}')
        # print('')

        return action

        # return tf.clip_by_value((self.actor_model(state) + tf.convert_to_tensor(self.noise(), dtype=tf.float32)).numpy()[0], self.env.action_space.low, self.env.action_space.high)


    # @tf.function
    def learn(self):
        # receives a batch of experiences from the replay buffer and learns from them
        
        # run callbacks on train batch begin
        if self.callback_list:
            self.callback_list.on_train_batch_begin(batch=self._step, logs=None)
        # sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # calculate critic loss and gradients
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_model(next_states)
            target_critic_values = tf.squeeze(self.target_critic_model((next_states, target_actions)), 1)
            # target_critic_values = self.target_critic_model((next_states, target_actions))
            #DEBUG
            # print(f'rewards: {rewards.shape}')
            # print(f'discount: {self.discount}')
            # print(f'target critic values shape: {target_critic_values.shape}')
            # print(f'dones: {dones.shape}')
            targets = rewards + self.discount * target_critic_values * (1 - dones)
            # print(f'targets: {targets}')
            # print(f'target shape: {targets.shape}')
            prediction = tf.squeeze(self.critic_model((states, actions)), 1)
            # prediction = self.critic_model((states, actions))
            critic_loss = self.critic_loss(targets, prediction)
            # tf.print(f'critic loss: {critic_loss}')
        
        # add critic loss to critic loss metric object
        # self.critic_loss_metric.update_state(critic_loss)
        critic_gradient = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_model.optimizer.apply_gradients(zip(critic_gradient, self.critic_model.trainable_variables))

        # calculate actor loss and gradients
        with tf.GradientTape() as tape:
            action_values = self.actor_model(states)
            values = self.critic_model((states, action_values))
            actor_loss = -tf.math.reduce_mean(values)
            # tf.print(f'actor loss: {actor_loss}')
        
        # # add actor loss to actor loss metric object
        # self.actor_loss(actor_loss)
        
        # update gradients
        actor_gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(zip(actor_gradient, self.actor_model.trainable_variables))

        # self.target_critic_model.set_weights([self.tau * cw + (1 - self.tau) * tcw for cw, tcw in zip(self.critic_model.get_weights(), self.target_critic_model.get_weights())])
        # self.target_actor_model.set_weights([self.tau * aw + (1 - self.tau) * taw for aw, taw in zip(self.actor_model.get_weights(), self.target_actor_model.get_weights())])
        
        
        # def update_target_network(current_weights, target_weights):
        #     for cw, tw in zip(current_weights, target_weights):
        #         # print('iterating over critic model weights')
        #         tw.assign(self.tau * cw + (1 - self.tau) * tw)

        # tf.py_function(update_target_network, [self.critic_model.variables, self.target_critic_model.variables], [], name='update_target_critic_model')
        # tf.py_function(update_target_network, [self.actor_model.variables, self.target_actor_model.variables], [], name='update_target_actor_model')

        # for cw, tw in zip(self.actor_model.variables, self.target_actor_model.variables):
        #     # print('iterating over actor model weights')
        #     tw.assign(self.tau * cw + (1 - self.tau) * tw)

        
        # define a tf.py_function inside tf.function graph to extract values for wandb logging if self._wandb
        # def update_wandb_metrics(temporal_difference, action_values, actor_loss, critic_loss, mean_policy_grad_mean,
        #                          mean_policy_grad_std, mean_policy_grad_max, mean_policy_grad_min, mean_policy_grad_norm,
        #                          mean_value_grad_mean, mean_value_grad_std, mean_value_grad_max, mean_value_grad_min, mean_value_grad_norm):
            
        #     self._train_step_config["temporal_difference"] = tf.reduce_mean(temporal_difference)
        #     mean_action_values = tf.reduce_mean(action_values, axis=0)
        #     for i, a in enumerate(mean_action_values):
        #         self._train_step_config[f"action_value {i}"] = a
            
        #     self._train_step_config["policy_loss"] = actor_loss
        #     self._train_step_config["value_loss"] = critic_loss
        #     self._train_step_config['mean_policy_grad_mean'] = mean_policy_grad_mean
        #     self._train_step_config['mean_policy_grad_std'] = mean_policy_grad_std
        #     self._train_step_config['mean_policy_grad_max'] = mean_policy_grad_max
        #     self._train_step_config['mean_policy_grad_min'] = mean_policy_grad_min
        #     self._train_step_config['mean_policy_grad_norm'] = mean_policy_grad_norm
        #     self._train_step_config['mean_value_grad_mean'] = mean_value_grad_mean
        #     self._train_step_config['mean_value_grad_std'] = mean_value_grad_std
        #     self._train_step_config['mean_value_grad_max'] = mean_value_grad_max
        #     self._train_step_config['mean_value_grad_min'] = mean_value_grad_min
        #     self._train_step_config['mean_value_grad_norm'] = mean_value_grad_norm
            
            
        # # log to wandb if using wandb callback
        # if self._wandb:
            
        #     mean_policy_grad_mean = tf.reduce_mean([tf.reduce_mean(g) for g in actor_gradient if g is not None])
        #     mean_policy_grad_std = tf.reduce_mean([tf.math.reduce_std(g) for g in actor_gradient if g is not None])
        #     mean_policy_grad_max = tf.reduce_mean([tf.reduce_max(g) for g in actor_gradient if g is not None])
        #     mean_policy_grad_min = tf.reduce_mean([tf.reduce_min(g) for g in actor_gradient if g is not None])
        #     mean_policy_grad_norm = tf.reduce_mean([tf.norm(g) for g in actor_gradient if g is not None])
        #     mean_value_grad_mean = tf.reduce_mean([tf.reduce_mean(g) for g in critic_gradient if g is not None])
        #     mean_value_grad_std = tf.reduce_mean([tf.math.reduce_std(g) for g in critic_gradient if g is not None])
        #     mean_value_grad_max = tf.reduce_mean([tf.reduce_max(g) for g in critic_gradient if g is not None])
        #     mean_value_grad_min = tf.reduce_mean([tf.reduce_min(g) for g in critic_gradient if g is not None])
        #     mean_value_grad_norm = tf.reduce_mean([tf.norm(g) for g in critic_gradient if g is not None])
        #     tf.py_function(update_wandb_metrics, [targets - prediction, self.actor_model(states), actor_loss, critic_loss,
        #                                          mean_policy_grad_mean, mean_policy_grad_std, mean_policy_grad_max,
        #                                          mean_policy_grad_min, mean_policy_grad_norm, mean_value_grad_mean,
        #                                          mean_value_grad_std, mean_value_grad_max, mean_value_grad_min,
        #                                          mean_value_grad_norm], [], name='WandB_update_metrics')
            # self._train_step_config['state_means'] = tf.reduce_mean(states, axis=0)

        # if self.callback_list:
        #     self._train_step_config = {
        #         k: v.numpy() if hasattr(v, 'numpy') else v for k, v in self._train_step_config.items()
        #     }
            

        # Reset metrics every step
        # self.actor_loss.reset_state()
        # self.critic_loss_metric.reset_state()
    
    # @tf.function
    def update_target_network(self, current_weights, target_weights):
        for cw, tw in zip(current_weights, target_weights):
            # print('iterating over critic model weights')
            tw.assign(self.tau * cw + (1 - self.tau) * tw)

    def train(
        self, num_episodes, render: bool = False, render_freq: int = None, save_dir=None):
        """Trains the model for 'episodes' number of episodes."""

        #DEBUG
        # print("ddpg train called...")
        
        if save_dir:
            self.save_dir = save_dir
        if self.callback_list:
            self.callback_list.on_train_begin(logs=self._config)
        # set tensorboard train log directory
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = f'tensorboard/logs/train/{current_time}'
        # create a summary writer to send data to tensorboard
        # self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, context='train')
        if self._wandb:
            # set step counter
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
            if self.callback_list:
                self.callback_list.on_epoch_begin(epoch=self._step, logs=None)
            # reset noise
            self.noise.reset()
            # reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0  # Initialize steps counter for the episode
            while not done:
                step_start_time = time.time()
                # action_time = time.time()
                action = self.get_action(state)
                # print(f"Action time: {time.time() - action_time}")
                next_state, reward, term, trunc, _ = self.env.step(action.numpy()) # might have to use action.numpy()
                step_time = time.time() - step_start_time
                step_time_history.append(step_time)
                # store trajectory in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                episode_steps += 1  # Increment steps counter for the episode
                
                # check if enough samples in replay buffer and if so, learn from experiences
                if self.replay_buffer.counter > self.batch_size:
                    learn_time = time.time()
                    self.learn()
                    self.update_target_network(self.critic_model.variables, self.target_critic_model.variables)
                    self.update_target_network(self.actor_model.variables, self.target_actor_model.variables)

                    learning_time_history.append(time.time() - learn_time)
                
                # log to wandb if using wandb callback
                if self._wandb:
                    self._train_step_config["step_reward"] = reward
                    self._train_step_config["step_time"] = step_time
                    self.callback_list.on_train_batch_end(batch=self._step, logs=self._train_step_config)
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

            self._train_episode_config["episode_reward"] = episode_reward
            self._train_episode_config["avg_reward"] = avg_reward
            self._train_episode_config["episode_time"] = episode_time
            # self._train_episode_config["avg_episode_time"] = avg_episode_time
            # self._train_episode_config["avg_step_time"] = avg_step_time
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                self.save()
            else:
                self._train_episode_config["best"] = False

            if self.callback_list:
                self.callback_list.on_epoch_end(
                    epoch=self._step, logs=self._train_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}, episode_time {episode_time:.2f}s, avg_episode_time {avg_episode_time:.2f}s, avg_step_time {avg_step_time:.6f}s, avg_learn_time {avg_learn_time:.6f}s, avg_steps_per_episode {avg_steps_per_episode:.2f}")

        if self.callback_list:
            self.callback_list.on_train_end(logs=self._train_episode_config)
        # close the environment
        self.env.close()

       
    def test(self, num_episodes, render, render_freq):
        """Runs a test over 'num_episodes'."""

        # set tensorboard test log directory
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.test_log_dir = f'tensorboard/logs/test/{current_time}'

        # instantiate list to store reward history
        reward_history = []
        # instantiate new environment
        self.env = self._initialize_env(render, render_freq, context='test')
        if self.callback_list:
            self.callback_list.on_test_begin(logs=self._config)

        step = 1
        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_test_batch_begin(batch=step, logs=None)
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
                step += 1
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])
            self._test_episode_config["episode_reward"] = episode_reward
            self._test_episode_config["avg_reward"] = avg_reward
            if self.callback_list:
                self.callback_list.on_test_batch_end(
                    batch=step, logs=self._test_episode_config
                )

            print(f"episode {i}, score {episode_reward}, avg_score {avg_reward}")

        if self.callback_list:
            self.callback_list.on_test_end(logs=self._test_episode_config)
        # close the environment
        self.env.close()

    def get_config(self):
        # return {
        #     "agent_type": self.__class__.__name__,
        #     "env": self.env.spec.id,
        #     # "actor_model": self.actor_model.__class__.__name__,
        #     # "critic_model": self.critic_model.__class__.__name__,
        #     "actor_model": self.actor_model.get_config(),
        #     "critic_model": self.critic_model.get_config(),
        #     "discount": self.discount,
        #     "tau": self.tau,
        #     "replay_buffer": self.replay_buffer.get_config(),
        #     "batch_size": self.batch_size,
        #     "noise": self.noise.get_config(),
        #     "callbacks": [callback.get_config() for callback in self.callback_list.callbacks],

        #     "save_dir": self.save_dir
        # }
        return {
                "agent_type": self.__class__.__name__,
                "env": self.env.spec.id,
                "actor_model": self.actor_model.model.get_config(),
                "critic_model": self.critic_model.model.get_config(),
                "discount": self.discount,
                "tau": self.tau,
                "replay_buffer": self.replay_buffer.get_config(),
                "batch_size": self.batch_size,
                "noise": self.noise.get_config(),
                "callbacks": [callback.get_config() for callback in self.callback_list.callbacks],

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
        #     for callback in self.callback_list:
        #         if isinstance(callback, tf_callbacks.WandbCallback):
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
        actor_model = models.ActorModel.load(folder, load_weights)
        # load value model
        critic_model = models.CriticModel.load(folder, load_weights)
        # load replay buffer
        obj_config['replay_buffer']['config']['env'] = gym.make(obj_config['env'])
        replay_buffer = helper.ReplayBuffer(**obj_config["replay_buffer"]["config"])
        # load noise
        noise = helper.Noise.create_instance(obj_config["noise"]["class_name"], **obj_config["noise"]["config"])
        # load wandb config if exists
        # if os.path.exists(Path(folder).joinpath(Path("wandb_config.json"))):
        #     callbacks = [
        #         tf_callbacks.WandbCallback.load(
        #             Path(folder).joinpath(Path("wandb_config.json"))
        #         )
        #     ]
        # else:
        #     callbacks = None
        
        # create callbacks list from obj_config['callbacks']
        # callbacks = []
        # for callback in obj_config['callbacks']:
        #     if callback == 'WandbCallback':
        #         callbacks.append(tf_callbacks.WandbCallback.load(
        #             Path(folder).joinpath(Path("wandb_config.json"))
        #         ))
        #     if callback == 'DashCallback':
        #         callbacks.append(dash_callbacks.DashCallback(obj_config['callbacks'][callback]['config']))
            # else:
            #     callbacks.append(helper.get_callback_instance(callback, obj_config['callbacks'][callback]['config']))

        callbacks = [tf_callbacks.load(callback_info['class_name'], callback_info['config']) for callback_info in obj_config['callbacks']]

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
                if isinstance(callback, tf_callbacks.WandbCallback):
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
