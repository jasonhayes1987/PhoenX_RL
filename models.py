"""Holds Model classes used for Reinforcement learning."""

# imports
import json
import os
from typing import List, Tuple
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras import Model, optimizers, initializers
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
    
class ActorModel(Model):
    """Actor model for predicting action values."""
    
    def __init__(
            self,
            env: gym.Env,
            hidden_layers: List[Tuple[int, str, initializers.Initializer]] = None,
            learning_rate: float = 0.0001,
            optimizer: str = "adam",
    ):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        if hidden_layers is None:
            hidden_layers = [(400, "relu", initializers.VarianceScaling(scale=1.0,
                                                                        mode='fan_in',
                                                                        distribution='uniform')),
                             (300, "relu", initializers.VarianceScaling(scale=1.0,
                                                                        mode='fan_in',
                                                                        distribution='uniform'))
                            ]
        self.hidden_layers = hidden_layers
        
        # self.model = keras.Sequential()
        # for size, activation, initializer in hidden_layers:
        #     self.model.add(
        #         Dense(size,
        #               activation=activation,
        #               kernel_initializer=initializer,
        #               bias_initializer=initializer)
        #     )
        # self.model.add(Dense(env.action_space.n,
        #                      activation='tanh',
        #                      kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
        #                      bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3)))

        # Switch to functional API
        self.input_layer = Input(shape=self.env.observation_space.shape)
        out = self.input_layer
        for units, activation, initializer in hidden_layers:
            out = Dense(units,
                        activation=activation,
                        kernel_initializer=initializer,
                        bias_initializer=initializer)(out)
        self.output_layer = Dense(env.action_space.shape[0],
                                  activation='tanh',
                                  kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                  bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(out)                               
        
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)
    
        try:
            self.optimizer = tf.keras.optimizers.get(optimizer)
        except ValueError as e:
            print(f"Optimizer '{optimizer}' not found. Using Adam instead. Original error: {e}")
            self.optimizer = optimizers.Adam(self.learning_rate)  # Setting Adam as the default optimizer

        self.build(np.expand_dims(env.observation_space.sample(), axis=0).shape)

    def call(self, state):
        """Forward Propogation."""
        return self.model(state)
    
    def _set_learning_rate(self, learning_rate):
        """Sets learning rate of optimizer."""
        self.optimizer.learning_rate = learning_rate

    def get_config(self):
        return {
            'env': self.env,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer
        }

    def save(self, folder):
        """Save model."""
        # makes directory if it doesn't exist
        os.makedirs(folder + "/policy_model", exist_ok=True)
        self.model.save(folder + "/policy_model")
        obj_config = {
        "env_name": self.env.spec.id,
        "hidden_layers": [
            (size, activation, {"class_name": initializer.__class__.__name__, "config": initializer.get_config()})
            for size, activation, initializer in self.hidden_layers
        ],
    }
        with open(folder + "/policy_model/obj_config.json", "w", encoding="utf-8") as f:
            json.dump((obj_config), f)
        opt_config = optimizers.serialize(self.optimizer)
        with open(
            folder + "/policy_model/optimizer_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump((opt_config), f)

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
            Path(folder).joinpath(Path("policy_model/optimizer_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            opt_config = json.load(f)
        opt = optimizers.deserialize(opt_config)
        # Reconstruct the initializers from the stored representation
        hidden_layers = [(size, activation, tf.keras.initializers.get(initializer))
                                        for size, activation, initializer in obj_config["hidden_layers"]
                                       ]
        actor_model = cls(
            env=gym.make(obj_config["env_name"]),
            hidden_layers=hidden_layers,
            optimizer=opt,
        )
        actor_model.model = keras.models.load_model(
            Path(folder).joinpath(Path("policy_model"))
        )
        return actor_model

class CriticModel(Model):
    """Critic model for predicting state/action values."""
    
    def __init__(
            self,
            env: gym.Env,
            state_layers: List[Tuple[int, str, initializers.Initializer]] = None,
            action_layers: List[Tuple[int, str, initializers.Initializer]] = None,
            merged_layers: List[Tuple[int, str, initializers.Initializer]] = None,
            learning_rate: float = 0.001,
            optimizer: str = "adam",
    ):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        # Default configurations if not provided
        if state_layers is None:
            state_layers = [(400, "relu", initializers.VarianceScaling(scale=1.0,
                                                                        mode='fan_in',
                                                                        distribution='uniform',
                                                                        seed=np.random.randint(0, 100))),
                            ]
        self.state_layers = state_layers
        if action_layers is None:
            action_layers = []
        self.action_layers = action_layers
        if merged_layers is None:
            merged_layers = [(300, "relu", initializers.VarianceScaling(scale=1.0,
                                                                        mode='fan_in',
                                                                        distribution='uniform',
                                                                        seed=np.random.randint(0, 100))),
                            ]
        self.merged_layers = merged_layers

        # State and Action Inputs
        self.state_input = Input(shape=self.env.observation_space.shape)
        self.action_input = Input(shape=self.env.action_space.shape)

        # State processing
        state_out = self.state_input
        for units, activation, initializer in state_layers:
            state_out = Dense(units,
                              activation=activation,
                              kernel_initializer=initializer,
                              bias_initializer=initializer)(state_out)

        # Action processing
        action_out = self.action_input
        # DEBUGGING
        print(f'action out: {action_out}')
        for units, activation, initializer in action_layers:
            action_out = Dense(units,
                               activation=activation,
                               kernel_initializer=initializer,
                               bias_initializer=initializer)(action_out)

        # Merge state and action
        merged = Concatenate()([state_out, action_out])

        # Post-merge layers
        for units, activation, initializer in merged_layers:
            merged = Dense(units,
                           activation=activation,
                           kernel_initializer=initializer,
                           bias_initializer=initializer)(merged)

        # Output layer
        self.output_layer = Dense(1,
                                  activation='relu',
                                  kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                  bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(merged)

        # Create the model
        self.model = tf.keras.Model(inputs=[self.state_input, self.action_input],
                                    outputs=self.output_layer)


        try:
            self.optimizer = tf.keras.optimizers.get(optimizer)
        except ValueError as e:
            print(f"Optimizer '{optimizer}' not found. Using Adam instead. Original error: {e}")
            self.optimizer = optimizers.Adam(self.learning_rate)  # Setting Adam as the default optimizer
        
        # compile model
        # self.model.compile(optimizer=self.optimizer)
        
        # run data through network to set shapes
        # self.build([tf.convert_to_tensor([self.env.observation_space.sample()], dtype=tf.float32).shape,
        #             tf.convert_to_tensor([self.env.action_space.sample()], dtype=tf.float32).shape])  
        self.build([np.expand_dims(env.observation_space.sample(), axis=0).shape,
                    np.expand_dims(env.action_space.sample(), axis=0).shape])

    def call(self, inputs):
        state, action = inputs
        return self.model([state, action])
    
    def _set_learning_rate(self, learning_rate):
        """Sets learning rate of optimizer."""
        self.optimizer.learning_rate = learning_rate

    def get_config(self):
        return {
            'env': self.env,
            'state_layers': self.state_layers,
            'action_layers': self.action_layers,
            'merged_layers': self.merged_layers,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer
        }

    def save(self, folder):
        """Save model."""
        # makes directory if it doesn't exist
        os.makedirs(folder + "/value_model", exist_ok=True)
        self.model.save(folder + "/value_model")
        obj_config = {
        "env_name": self.env.spec.id,
        "state_layers": [
            (size, activation, {"class_name": initializer.__class__.__name__, "config": initializer.get_config()})
            for size, activation, initializer in self.state_layers
        ],
        "action_layers": [
            (size, activation, {"class_name": initializer.__class__.__name__, "config": initializer.get_config()})
            for size, activation, initializer in self.action_layers
        ],
        "merged_layers": [
            (size, activation, {"class_name": initializer.__class__.__name__, "config": initializer.get_config()})
            for size, activation, initializer in self.merged_layers
        ],
        "learning_rate": self.learning_rate,
    }
        with open(folder + "/value_model/obj_config.json", "w", encoding="utf-8") as f:
            json.dump((obj_config), f)
        opt_config = optimizers.serialize(self.optimizer)
        with open(
            folder + "/value_model/optimizer_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump((opt_config), f)

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
            Path(folder).joinpath(Path("value_model/optimizer_config.json")),
            "r",
            encoding="utf-8",
        ) as f:
            opt_config = json.load(f)
        opt = optimizers.deserialize(opt_config)
        # Reconstruct the initializers from the stored representation
        state_layers = [(size, activation, tf.keras.initializers.get(initializer))
                        for size, activation, initializer in obj_config["state_layers"]
                       ]
        action_layers = [(size, activation, tf.keras.initializers.get(initializer))
                        for size, activation, initializer in obj_config["action_layers"]
                       ]
        merged_layers = [(size, activation, tf.keras.initializers.get(initializer))
                        for size, activation, initializer in obj_config["merged_layers"]
                       ]
        critic_model = cls(
            env=gym.make(obj_config["env_name"]),
            state_layers=state_layers,
            action_layers=action_layers,
            merged_layers=merged_layers,
            learning_rate=obj_config["learning_rate"],
            optimizer=opt,
        )
        critic_model.model = keras.models.load_model(
            Path(folder).joinpath(Path("value_model"))
        )
        return critic_model


def build_layers(units_per_layer: List[int], activation: str):
    """formats sweep_config into policy and value layers"""
    # get policy layers
    layers = []
    for units in units_per_layer:
        layers.append((units, activation))
    return layers
