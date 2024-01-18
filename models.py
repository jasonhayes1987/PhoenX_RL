"""Holds Model classes used for Reinforcement learning."""

# imports
import json
import os
from typing import List, Tuple
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, optimizers
from tensorflow.keras.initializers import HeNormal
import gymnasium as gym
import numpy as np


class PolicyModel(Model):
    """Policy model for predicting probability distribution of actions.

    Attributes:
      env: OpenAI gym environment.
      hidden_layers: List of hidden layer sizes and activations.
      optimizer: Optimizer for training.

    """

    def __init__(
        self,
        env: gym.Env,
        hidden_layers: List[Tuple[int, str]] = None,
        optimizer: keras.optimizers = optimizers.Adam(0.001),
    ):
        super().__init__()
        self.env = env
        if hidden_layers is None:
            hidden_layers = [(100, "relu")]
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.model = keras.Sequential()
        for size, activation in hidden_layers:
            self.model.add(
                Dense(size, activation=activation, kernel_initializer=HeNormal())
            )
        self.model.add(
            Dense(env.action_space.n, activation=None, kernel_initializer=HeNormal())
        )
        self.build(np.expand_dims(env.observation_space.sample(), axis=0).shape)

    # @tf.function()
    def call(self, state, return_logits=False):
        """Forward Propogation."""
        logits = self.model(state)
        if return_logits:
            return logits
        return tf.nn.softmax(logits)

    def _set_learning_rate(self, learning_rate):
        """Sets learning rate of optimizer."""
        self.optimizer.learning_rate = learning_rate

    def save(self, folder):
        """Save model."""
        # makes directory if it doesn't exist
        os.makedirs(folder + "/policy_model", exist_ok=True)
        self.model.save(folder + "/policy_model")
        obj_config = {
            "env_name": self.env.spec.id,
            "hidden_layers": self.hidden_layers,
        }
        with open(folder + "/policy_model/obj_config.json", "w", encoding="utf-8") as f:
            json.dump((obj_config), f)
        policy_opt_config = optimizers.serialize(self.optimizer)
        with open(
            folder + "/policy_model/policy_optimizer_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump((policy_opt_config), f)

    @classmethod
    def load(cls, folder):
        """Load model."""
        with open(
            Path(folder).joinpath(Path("policy_model/obj_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            obj_config = json.load(f)
        with open(
            Path(folder).joinpath(Path("policy_model/policy_optimizer_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            policy_opt_config = json.load(f)
        policy_opt = optimizers.deserialize(policy_opt_config)
        policy_model = cls(
            env=gym.make(obj_config["env_name"]),
            hidden_layers=obj_config["hidden_layers"],
            optimizer=policy_opt,
        )
        policy_model.model = keras.models.load_model(
            Path(folder).joinpath(Path("policy_model"))
        )
        return policy_model


class ValueModel(Model):
    """Value model for predicting state values.

    Attributes:
      env: OpenAI gym environment.
      hidden_layers: List of hidden layer sizes and activations.
      optimizer: Optimizer for training.

    """

    def __init__(
        self,
        env: gym.Env,
        hidden_layers: List[Tuple[int, str]] = None,
        optimizer: keras.optimizers = optimizers.Adam(0.001),
    ):
        super().__init__()
        self.env = env
        if hidden_layers is None:
            hidden_layers = [(100, "relu")]
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.model = keras.Sequential()
        for size, activation in hidden_layers:
            self.model.add(
                Dense(size, activation=activation, kernel_initializer=HeNormal())
            )
        self.model.add(Dense(1, activation=None, kernel_initializer=HeNormal()))
        self.build(np.expand_dims(env.observation_space.sample(), axis=0).shape)

    # @tf.function()
    def call(self, state):
        """Forward propogation."""

        return self.model(state)

    def _set_learning_rate(self, learning_rate):
        """Sets learning rate of optimizer."""
        self.optimizer.learning_rate = learning_rate

    def save(self, folder):
        """Save model."""
        os.makedirs(folder + "/value_model", exist_ok=True)
        self.model.save(folder + "/value_model")
        obj_config = {
            "env_name": self.env.spec.id,
            "hidden_layers": self.hidden_layers,
        }
        with open(folder + "/value_model/obj_config.json", "w", encoding="utf-8") as f:
            json.dump((obj_config), f)
        value_opt_config = optimizers.serialize(self.optimizer)
        with open(
            folder + "/value_model/value_optimizer_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump((value_opt_config), f)

    @classmethod
    def load(cls, folder):
        """Load model."""
        with open(
            Path(folder).joinpath(Path("value_model/obj_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            obj_config = json.load(f)
        with open(
            Path(folder).joinpath(Path("value_model/value_optimizer_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            value_opt_config = json.load(f)
        value_opt = optimizers.deserialize(value_opt_config)
        value_model = cls(
            env=gym.make(obj_config["env_name"]),
            hidden_layers=obj_config["hidden_layers"],
            optimizer=value_opt,
        )
        value_model.model = keras.models.load_model(
            Path(folder).joinpath(Path("value_model"))
        )
        return value_model


def build_layers(units_per_layer: List[int], activation: str):
    """formats sweep_config into policy and value layers"""
    # get policy layers
    layers = []
    for units in units_per_layer:
        layers.append((units, activation))
    return layers
