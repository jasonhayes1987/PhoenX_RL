"""This module holds the Agent base class and all RL agents as subclasses  It also 
provides helper functions for loading any subclass of type Agent."""

# imports
import json
import os
from typing import List
from pathlib import Path

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
                if isinstance(callback, wandb_support.WandbCallback):
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
            env, hidden_layers=policy_layers, optimizer=policy_optimizer
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
            env = gym.make(self.env.spec.id, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                env,
                self.save_dir + "/renders",
                episode_trigger=lambda episode_id: episode_id % render_freq == 0,
            )

        return gym.make(self.env.spec.id)

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

    def save(self):
        """Saves the model."""
        obj_config = {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "discount": self.discount,
            "learning_rate": self.learning_rate,
            "policy_trace_decay": self.policy_trace_decay,
            "value_trace_decay": self.value_trace_decay,
            "callbacks": [callback.__class__.__name__ for callback in self.callbacks],
            "save_dir": self.save_dir,
        }

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
                if isinstance(callback, wandb_support.WandbCallback):
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
                wandb_support.WandbCallback.load(
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
            if isinstance(callback, wandb_support.WandbCallback):
                agent._wandb = True
                agent._train_config = {}
                agent._train_episode_config = {}
                agent._train_step_config = {}
                agent._test_config = {}

        return agent

    # Agent class


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
                if isinstance(callback, wandb_support.WandbCallback):
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
            env, hidden_layers=policy_layers, optimizer=policy_optimizer
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
            env = gym.make(self.env.spec.id, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                env,
                self.save_dir + "/renders",
                episode_trigger=lambda episode_id: episode_id % render_freq == 0,
            )

        return gym.make(self.env.spec.id)

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

    def save(self):
        """Saves the model."""
        obj_config = {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "discount": self.discount,
            "learning_rate": self.learning_rate,
            "callbacks": [callback.__class__.__name__ for callback in self.callbacks],
            "save_dir": self.save_dir,
        }

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
                if isinstance(callback, wandb_support.WandbCallback):
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
                wandb_support.WandbCallback.load(
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
            if isinstance(callback, wandb_support.WandbCallback):
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
        self.target_actor_model = self.clone_model(actor_model)
        self.target_critic_model = self.clone_model(critic_model)
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
        self.callbacks = callbacks
        self.save_dir = save_dir
        self._DEBUG = _DEBUG
        # instantiate and set keras loss objects
        # self.actor_loss = tf.keras.metrics.Mean(name="Actor Loss")
        # self.critic_loss = tf.keras.metrics.Mean(name="Critic Loss")
        self.critic_loss = tf.keras.losses.MeanSquaredError()
        if callbacks:
            self.callback_list = self._create_callback_list(callbacks)
            for callback in self.callback_list:
                if isinstance(callback, wandb_support.WandbCallback):
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
        # Retrieve the model's configuration
        config = model.get_config()

        # Create a new model instance with the same configuration
        cloned_model = model.__class__(**config)

        # Copy the weights from the original model
        cloned_model.set_weights(model.get_weights())

        return cloned_model
    
    def build(self):
        pass

    def _create_callback_list(self, callbacks):
        if callbacks is None:
            callbacks = []
        callback_list = CallbackList(callbacks)

        return callback_list

    def _initialize_env(self, render=False, render_freq=10):
        """Initializes a new environment."""
        if render:
            env = gym.make(self.env.spec.id, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                env,
                self.save_dir + "/renders",
                episode_trigger=lambda episode_id: episode_id % render_freq == 0,
            )

        return gym.make(self.env.spec.id)
    
    def get_action(self, state):
        # receives current state and returns a vector of action values from policy model
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return (self.actor_model(state) + tf.convert_to_tensor(self.noise(), dtype=tf.float32)).numpy()[0] * self.env.action_space.high

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
        # rewards = tf.convert_to_tensor(np.expand_dims(rewards, axis=1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        # dones = tf.convert_to_tensor(np.expand_dims(dones, axis=1), dtype=tf.float32)
        # dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        if self._DEBUG:
            print(f'states shape: {states.shape}')
            print(f'states: {states}')
            print(f'actions shape: {actions.shape}')
            print(f'actions: {actions}')
            print(f'rewards shape: {rewards.shape}')
            print(f'rewards: {rewards}')
            print(f'dones shape: {dones.shape}')
            print(f'dones: {dones}')
        
        # calculate critic loss and gradients
        with tf.GradientTape() as tape:
            # set target values using the target actor and critic models
            # targets = rewards + self.discount * self.target_critic_model([next_states, self.target_actor_model(next_states)]) * (1 - dones)
            target_actions = self.target_actor_model(next_states)
            if self._DEBUG:
                print(f'target action values shape: {target_actions.shape}')
                print(f'target action values: {target_actions}')
            target_critic_values = tf.squeeze(self.target_critic_model((next_states, target_actions)), 1)
            if self._DEBUG:
                print(f'target critic values shape: {target_critic_values.shape}')
                print(f'target critic values: {target_critic_values}')
            targets = rewards + self.discount * target_critic_values * (1-dones)
            if self._DEBUG:
                print(f'targets shape: {targets.shape}')
                print(f'targets: {targets}')
            prediction = tf.squeeze(self.critic_model((states, actions)),1)
            if self._DEBUG:
                print(f'predictions shape: {prediction.shape}')
                print(f'predictions: {prediction}')
            # calculate critic loss
            critic_loss = self.critic_loss(targets, prediction)
        
        if self._DEBUG:
            print(f'critic loss shape: {critic_loss.shape}')    
            print(f'critic loss: {critic_loss}')
        # calculate gradients
        critic_gradient = tape.gradient(critic_loss, self.critic_model.model.trainable_variables)
        if self._DEBUG:
            print(f'critic gradient: {critic_gradient}')
        # apply gradients to critic model
        self.critic_model.optimizer.apply_gradients(zip(critic_gradient, self.critic_model.model.trainable_variables))

        # calculate actor loss and gradients
        with tf.GradientTape() as tape:
            # get actions from actor model
            action_values = self.actor_model(states)
            if self._DEBUG:
                print(f'action values shape: {action_values.shape}')
                print(f'action values: {action_values}')
            values = self.critic_model((states, action_values))
            if self._DEBUG:
                print(f'values shape: {values.shape}')
                print(f'values: {values}')
            actor_loss = -tf.math.reduce_mean(values)
        
        if self._DEBUG:
            print(f'actor loss shape: {actor_loss.shape}')
            print(f'actor loss: {actor_loss}')
        # calculate gradients
        actor_gradient = tape.gradient(actor_loss, self.actor_model.model.trainable_variables)
        if self._DEBUG:
            print(f'actor gradient: {actor_gradient}')
        # apply gradients to actor model
        self.actor_model.optimizer.apply_gradients(zip(actor_gradient, self.actor_model.model.trainable_variables))

        # update target actor and critic models using soft update
        
        self.target_critic_model.set_weights([self.tau * cw + (1 - self.tau) * tcw for cw, tcw in zip(self.critic_model.get_weights(), self.target_critic_model.get_weights())])
        self.target_actor_model.set_weights([self.tau * aw + (1 - self.tau) * taw for aw, taw in zip(self.actor_model.get_weights(), self.target_actor_model.get_weights())])

        # log to wandb if using wandb callback
        if self._wandb: 
            self._train_step_config["temporal_difference"] = tf.reduce_mean(targets - prediction)
            for i, value in enumerate(tf.reduce_mean(action_values, axis=0)):
                self._train_step_config[f"action_value_{i}"] = value
            self._train_step_config["policy_loss"] = actor_loss
            self._train_step_config["value_loss"] = critic_loss.numpy()
            for p_grad, v_grad in zip(actor_gradient, critic_gradient):
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
                for e, s in enumerate(tf.reduce_mean(states, axis=0)):
                    self._train_step_config[f"state_{e}"] = s

        if self.callback_list:
            self.callback_list.on_train_batch_end(
                batch=self._step, logs=self._train_step_config
            )

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
        # set best reward
        best_reward = self.env.reward_range[0]
        # instantiate list to store reward history
        reward_history = []
        for i in range(num_episodes):
            if self.callback_list:
                self.callback_list.on_epoch_begin(epoch=self._step, logs=None)
            # reset noise
            self.noise.reset()
            # reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action) # might have to use action.numpy()
                # store trajectory in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                if term or trunc:
                    done = True
                episode_reward += reward
                state = next_state
                # log to wandb if using wandb callback
                if self._wandb:
                    self._train_step_config["action"] = action
                    self._train_step_config["step_reward"] = reward
                    if not done:
                        self._step += 1
                # check if enough samples in replay buffer and if so, learn from experiences
                if self.replay_buffer.counter > self.batch_size:
                    self.learn()
            
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-100:])

            self._train_episode_config["episode_reward"] = episode_reward
            self._train_episode_config["avg_reward"] = avg_reward
            # check if best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._train_episode_config["best"] = True
                # save model
                # self.save()
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

        
    def test(self,):
        pass

    def get_config(self):
        return {
                "env": self.env,
                "actor_model": self.actor_model,
                "critic_model": self.critic_model,
                "discount": self.discount,
                "tau": self.tau,
                "replay_buffer": self.replay_buffer,
                "batch_size": self.batch_size,
                "noise": self.noise,
                "callbacks": [callback for callback in self.callbacks],
                "save_dir": self.save_dir
    }

    def save(self):
        """Saves the model."""
        obj_config = {
            "agent_type": self.__class__.__name__,
            "env": self.env.spec.id,
            "actor_model": self.actor_model.__class__.__name__,
            "critic_model": self.critic_model.__class__.__name__,
            "discount": self.discount,
            "tau": self.tau,
            "replay_buffer": self.replay_buffer.__class__.__name__,
            "batch_size": self.batch_size,
            "noise": self.noise.__class__.__name__,
            "callbacks": [callback.__class__.__name__ for callback in self.callbacks],
            "save_dir": self.save_dir
        }

        # makes directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # writes and saves JSON file of reinforce agent config
        with open(self.save_dir + "/obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

        # saves policy and value model
        self.actor_model.save(self.save_dir)
        self.critic_model.save(self.save_dir)

        # if wandb callback, save wandb config
        if self._wandb:
            for callback in self.callback_list:
                if isinstance(callback, wandb_support.WandbCallback):
                    callback.save(self.save_dir + "/wandb_config.json")

    @classmethod
    def load(cls, folder: str = "models"):
        pass


def load_agent_from_config(config_path):
    """Loads an agent from a config file."""
    with open(
        Path(config_path).joinpath(Path("obj_config.json")), "r", encoding="utf-8"
    ) as f:
        config = json.load(f)

    agent_type = config["agent_type"]

    # Use globals() to get a reference to the class
    agent_class = globals().get(agent_type)

    if agent_class:
        return agent_class.load(config_path)

    raise ValueError(f"Unknown agent type: {agent_type}")


def get_agent_class_from_type(agent_type: str):
    """Builds an agent from a passed agent type str."""

    types = {"Actor Critic": "ActorCritic", "Reinforce": "Reinforce"}

    # Use globals() to get a reference to the class
    agent_class = globals().get(types[agent_type])

    if agent_class:
        return agent_class

    raise ValueError(f"Unknown agent type: {agent_type}")
