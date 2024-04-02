"""Holds Model classes used for Reinforcement learning."""

# imports
import json
import os
from typing import List, Tuple
from pathlib import Path
# import time

import torch
import torch.nn as nn
from torch import optim

import gymnasium as gym
import numpy as np
import cnn_models
import torch_utils


class Model(nn.Module):
    """Base class for all RL models."""
    def __init__(self):
        super().__init__()
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def _init_weights(self, module_dict, layer_config, prefix: str = ''):
        config_index = 0

        for layer_name, layer in module_dict.items():
            if 'activation' not in layer_name:
                _, _, init_config = layer_config[config_index]

                ##DEBUG
                print(f'layer {prefix}_dense_{config_index} using {init_config} for {layer}')
                
                if isinstance(init_config, dict):
                    if init_config['variance scaling']:
                        print('dict variance scaling')
                        torch_utils.VarianceScaling_(layer.weight, **init_config['variance scaling'])
                    
                    elif init_config['uniform']:
                        nn.init.uniform_(layer.weight, **init_config['uniform'])
                        nn.init.uniform_(layer.bias, **init_config['uniform'])

                    elif init_config['normal']:
                        nn.init.normal_(layer.weight, **init_config['normal'])
                        nn.init.normal_(layer.bias, **init_config['normal'])

                    elif init_config['constant']:
                        nn.init.constant_(layer.weight, **init_config['constant'])
                        nn.init.constant_(layer.bias, **init_config['constant'])

                    elif init_config['xavier uniform']:
                        nn.init.xavier_uniform_(layer.weight, **init_config['xavier uniform'])

                    elif init_config['xavier normal']:
                        nn.init.xavier_normal_(layer.weight, **init_config['xavier normal'])

                    elif init_config['kaiming uniform']:
                        nn.init.kaiming_uniform_(layer.weight, **init_config['kaiming uniform'])
                    
                    elif init_config['kaiming normal']:
                        nn.init.kaiming_normal_(layer.weight, **init_config['kaiming normal'])

                    elif init_config['truncated normal']:
                        nn.init.trunc_normal_(layer.weight, **init_config['truncated normal'])
                        nn.init.trunc_normal_(layer.bias, **init_config['truncated normal'])
                
                elif isinstance(init_config, str):
                    if init_config == 'variance scaling':
                        torch_utils.VarianceScaling_(layer.weight)

                    elif init_config == 'uniform':
                        nn.init.uniform_(layer.weight)
                        nn.init.uniform_(layer.bias)

                    elif init_config == 'normal':
                        nn.init.normal_(layer.weight)
                        nn.init.normal_(layer.bias)

                    elif init_config == 'ones':
                        nn.init.ones_(layer.weight)
                        nn.init.ones_(layer.bias)

                    elif init_config == 'zeros':
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)

                    elif init_config == 'xavier uniform':
                        nn.init.xavier_normal_(layer.weight)

                    elif init_config == 'xavier normal':
                        nn.init.xavier_normal_(layer.weight)

                    elif init_config == 'kaiming uniform':
                        nn.init.kaiming_uniform_(layer.weight)

                    elif init_config == 'kaiming normal':
                        nn.init.kaiming_normal_(layer.weight)
                    
                    elif init_config == 'truncated normal':
                        nn.init.trunc_normal_(layer.weight)
                        nn.init.trunc_normal_(layer.bias)
                
                else:
                    raise ValueError(f"Invalid init_config {init_config} index {config_index}")
                
                config_index += 1


    def _init_optimizer(self):
        if self.optimizer_class == 'Adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_class =='SGD':
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_class == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_class == 'Adagrad':
            return optim.Adagrad(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError


    def forward(self):
        pass


    def get_config(self):
        pass


    def save(self, folder):
        pass


    @classmethod
    def load(cls, folder):
        pass


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
        dense_layers: List[Tuple[int, str, str]] = None,
        optimizer: str = "Adam",
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.env = env
        self.layer_config = dense_layers
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        # self.optimizer.learning_rate = self.learning_rate

        # # Set the device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = device

        self.dense_layers = nn.ModuleDict()

        # set initial input size
        input_size = np.prod(env.observation_space.shape)

        # build model
        for i, (units, activation, _) in enumerate(self.layer_config):
            self.dense_layers[f'policy_dense_{i}'] = nn.Linear(input_size, units)
            
            if activation == 'relu':
                self.dense_layers[f'policy_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.dense_layers[f'policy_activation_{i}'] = nn.Tanh()
            
            # update input size to be output size of previous layer
            input_size = units

        # initialize dense layer weights
        self._init_weights(self.dense_layers, self.layer_config, 'policy')

        # add output layers to dict
        self.dense_layers['policy_output'] = nn.Linear(input_size, env.action_space.n)
        # self.dense_layers['policy_activation'] = nn.Softmax(dim=-1)
        
        # initialize weights of policy_output layer
        nn.init.kaiming_normal_(self.dense_layers['policy_output'].weight)
        # nn.init.zeros_(self.dense_layers['policy_output'].bias)

        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Move the model to the specified device
        self.to(self.device) 

    # def _init_weights(self):
    #     for i, (_, _, init_config) in enumerate(self.layer_config):
    #         layer = self.dense_layers[f'policy_dense_{i}']
    #         # for (name, layer) in self.dense_layers.items():

    #         ##DEBUG
    #         print(f'layer policy_dense_{i} using {init_config} for {layer}')

    #         if init_config == 'variance scaling':
    #             #DEBUG
    #             torch_utils.VarianceScaling_(layer.weight, **init_config['variance scaling'])
    #             # torch_utils.VarianceScaling_(layer.bias, **init_config['variance scaling'])
            
    #         if init_config == 'uniform':
    #             nn.init.uniform_(layer.weight, **init_config['uniform'])
    #             nn.init.uniform_(layer.bias, **init_config['uniform'])

    #         if init_config == 'normal':
    #             nn.init.normal_(layer.weight, **init_config['normal'])
    #             nn.init.normal_(layer.bias, **init_config['normal'])

    #         if init_config == 'constant':
    #             nn.init.constant_(layer.weight, **init_config['constant'])
    #             nn.init.constant_(layer.bias, **init_config['constant'])

    #         if init_config == 'ones':
    #             nn.init.ones_(layer.weight)
    #             nn.init.ones_(layer.bias)

    #         if init_config == 'zeros':
    #             nn.init.zeros_(layer.weight)
    #             nn.init.zeros_(layer.bias)

    #         if init_config == 'xavier uniform':
    #             nn.init.xavier_uniform_(layer.weight, **init_config['xavier uniform'])
    #             # nn.init.uniform_(layer.bias)

    #         if init_config == 'xavier normal':
    #             nn.init.xavier_normal_(layer.weight, **init_config['xavier normal'])
    #             # nn.init.normal_(layer.bias)

    #         if init_config == 'kaiming uniform':
    #             nn.init.kaiming_uniform_(layer.weight, **init_config['kaiming uniform'])
    #             # nn.init.uniform_(layer.bias)

    #         if init_config == 'kaiming normal':
    #             nn.init.kaiming_normal_(layer.weight, **init_config['kaiming normal'])
    #             # nn.init.normal_(layer.bias)

    #         if init_config == 'truncated normal':
    #             nn.init.trunc_normal_(layer.weight, **init_config['truncated normal'])
    #             nn.init.trunc_normal_(layer.bias, **init_config['truncated normal'])


    # def _init_optimizer(self):
    #     if self.optimizer_class == 'Adam':
    #         return optim.Adam(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class =='SGD':
    #         return optim.SGD(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'RMSprop':
    #         return optim.RMSprop(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'Adagrad':
    #         return optim.Adagrad(self.parameters(), lr=self.learning_rate)
    #     else:
    #         raise NotImplementedError


    def forward(self, x):
        """Forward Propogation."""
        x = x.to(self.device)
        for layer in self.dense_layers.values():
            x = layer(x)
        
        return x
    

    # def _set_learning_rate(self, learning_rate):
    #     """Sets learning rate of optimizer."""
    #     self.optimizer.learning_rate = learning_rate

    def get_config(self):
        """Get model config."""

        config = {
            'env': self.env.spec.id,
            'hidden_layers': len(self.dense_layers),
            'dense_layers': self.layer_config,
            'optimizer': self.optimizer_class,
            'learning_rate': self.learning_rate,
        }

        return config

    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "policy_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')

        obj_config = {
            "env_name": self.env.spec.id,
            "dense_layers": self.layer_config,
            "optimizer": self.optimizer_class,
            "learning_rate": self.learning_rate,
        }

        with open(model_dir / "obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)

    @classmethod
    def load(cls, folder):
        model_dir = Path(folder) / "policy_model"
        obj_config_path = model_dir / "obj_config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if obj_config_path.is_file():
            with open(obj_config_path, "r", encoding="utf-8") as f:
                obj_config = json.load(f)
            env = obj_config.get("env_name", "Custom/UnknownEnv")
            dense_layers = obj_config.get("dense_layers", [])
            optimizer = obj_config.get("optimizer", "Adam")
            learning_rate = obj_config.get("learning_rate", 0.0001)
        else:
            raise FileNotFoundError(f"No configuration file found in {obj_config_path}")

        model = cls(env, dense_layers, optimizer, learning_rate)
        model.load_state_dict(torch.load(model_path))

        return model


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
        dense_layers: List[Tuple[int, str, str]] = None,
        optimizer: str = 'Adam',
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.env = env
        self.layer_config = dense_layers
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        # # Set the device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = device

        self.dense_layers = nn.ModuleDict()
        
        # set initial input size
        input_size = np.prod(env.observation_space.shape)

        # build model
        for i, (units, activation, _) in enumerate(self.layer_config):
            self.dense_layers[f'value_dense_{i}'] = nn.Linear(input_size, units)
            
            if activation == 'relu':
                self.dense_layers[f'value_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.dense_layers[f'value_activation_{i}'] = nn.Tanh()
            
            # update input size to be output size of previous layer
            input_size = units
        
        # initialize dense layer weights
        self._init_weights(self.dense_layers, self.layer_config, 'value')

        # add output layers to dict
        self.dense_layers['value_output'] = nn.Linear(input_size, 1)

        # initialize weights of policy_output layer
        nn.init.kaiming_normal_(self.dense_layers['value_output'].weight)
        # nn.init.zeros_(self.dense_layers['value_output'].bias)
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Move the model to the specified device
        self.to(self.device)

    # def _init_weights(self):
    #     for i, (_, _, init_config) in enumerate(self.layer_config):
    #         layer = self.dense_layers[f'value_dense_{i}']
    #         # for (name, layer) in self.dense_layers.items():

    #         ##DEBUG
    #         print(f'layer value_dense_{i} using {init_config} for {layer}')

    #         if init_config == 'variance scaling':
    #             if isinstance(init_config, dict):
    #                 torch_utils.VarianceScaling_(layer.weight, **init_config['variance scaling'])
    #             elif isinstance(init_config, str):
    #                 torch_utils.VarianceScaling_(layer.weight, **init_config)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")
                
            
    #         if init_config == 'uniform':
    #             if isinstance(init_config, dict):
    #                 nn.init.uniform_(layer.weight, **init_config['uniform'])
    #                 nn.init.uniform_(layer.bias, **init_config['uniform'])
    #             elif isinstance(init_config, str):
    #                 nn.init.uniform_(layer.weight, **init_config)
    #                 nn.init.uniform_(layer.bias, **init_config)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'normal':
    #             if isinstance(init_config, dict):
    #                 nn.init.normal_(layer.weight, **init_config['normal'])
    #                 nn.init.normal_(layer.bias, **init_config['normal'])
    #             elif isinstance(init_config, str):
    #                 nn.init.normal_(layer.weight, **init_config)
    #                 nn.init.normal_(layer.bias, **init_config)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'constant':
    #             if isinstance(init_config, dict):
    #                 nn.init.constant_(layer.weight, **init_config['constant'])
    #                 nn.init.constant_(layer.bias, **init_config['constant'])
    #             elif isinstance(init_config, str):
    #                 nn.init.constant_(layer.weight, **init_config)
    #                 nn.init.constant_(layer.bias, **init_config)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'ones':
    #             nn.init.ones_(layer.weight)
    #             nn.init.ones_(layer.bias)

    #         if init_config == 'zeros':
    #             nn.init.zeros_(layer.weight)
    #             nn.init.zeros_(layer.bias)

    #         if init_config == 'xavier uniform':
    #             if isinstance(init_config, dict):
    #                 nn.init.xavier_uniform_(layer.weight, **init_config['xavier uniform'])
    #             elif isinstance(init_config, str):
    #                 nn.init.xavier_uniform_(layer.weight)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'xavier normal':
    #             if isinstance(init_config, dict):
    #                 nn.init.xavier_normal_(layer.weight, **init_config['xavier normal'])
    #             elif isinstance(init_config, str):
    #                 nn.init.xavier_normal_(layer.weight)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'kaiming uniform':
    #             if isinstance(init_config, dict):
    #                 nn.init.kaiming_uniform_(layer.weight, **init_config['kaiming uniform'])
    #             elif isinstance(init_config, str):
    #                 nn.init.kaiming_uniform_(layer.weight)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")
                
    #         if init_config == 'kaiming normal':
    #             if isinstance(init_config, dict):
    #                 nn.init.kaiming_normal_(layer.weight, **init_config['kaiming normal'])
    #             elif isinstance(init_config, str):
    #                 nn.init.kaiming_normal_(layer.weight)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")

    #         if init_config == 'truncated normal':
    #             if isinstance(init_config, dict):
    #                 nn.init.trunc_normal_(layer.weight, **init_config['truncated normal'])
    #                 nn.init.trunc_normal_(layer.bias, **init_config['truncated normal'])
    #             elif isinstance(init_config, str):
    #                 nn.init.trunc_normal_(layer.weight)
    #                 nn.init.trunc_normal_(layer.bias)
    #             else:
    #                 raise TypeError(f"Invalid init_config type {type(init_config)}")
            
    #         else:
    #             raise ValueError(f"Invalid init_config {init_config}")


    # def _init_optimizer(self):
    #     if self.optimizer_class == 'Adam':
    #         return optim.Adam(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class =='SGD':
    #         return optim.SGD(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'RMSprop':
    #         return optim.RMSprop(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'Adagrad':
    #         return optim.Adagrad(self.parameters(), lr=self.learning_rate)
    #     else:
    #         raise NotImplementedError
    
    
    def forward(self, x):
        """Forward Propogation."""
        x = x.to(self.device)
        for layer in self.dense_layers.values():
            x = layer(x)
        
        return x


    def get_config(self):
        """Get model config."""

        config = {
            'env': self.env.spec.id,
            'hidden_layers': len(self.dense_layers),
            'dense_layers': self.layer_config,
            'optimizer': self.optimizer_class,
            'learning_rate': self.learning_rate,
        }

        return config


    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "value_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')

        obj_config = {
            "env_name": self.env.spec.id,
            "dense_layers": self.layer_config,
            "optimizer": self.optimizer_class,
            "learning_rate": self.learning_rate,
        }

        with open(model_dir / "obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)


    @classmethod
    def load(cls, folder):
        model_dir = Path(folder) / "value_model"
        obj_config_path = model_dir / "obj_config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if obj_config_path.is_file():
            with open(obj_config_path, "r", encoding="utf-8") as f:
                obj_config = json.load(f)
            env = obj_config.get("env_name", "Custom/UnknownEnv")
            dense_layers = obj_config.get("dense_layers", [])
            optimizer = obj_config.get("optimizer", "Adam")
            learning_rate = obj_config.get("learning_rate", 0.0001)
        else:
            raise FileNotFoundError(f"No configuration file found in {obj_config_path}")

        model = cls(env, dense_layers, optimizer, learning_rate)
        model.load_state_dict(torch.load(model_path))

        return model


class ActorModel(Model):
    def __init__(self, env, cnn_model=None, dense_layers=None, optimizer: str = 'Adam', learning_rate=0.0001):
        super().__init__()
        self.env = env
        self.layer_config = dense_layers
        self.cnn_model = cnn_model
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        
        self.dense_layers = nn.ModuleDict()
        
        if self.cnn_model:
            obs_shape = env.observation_space.shape
            dummy_input = torch.zeros(1, *obs_shape, device=self.device)
            dummy_input = dummy_input.permute(0, 3, 1, 2)
            cnn_output = self.cnn_model(dummy_input)
            input_size = cnn_output.view(cnn_output.size(0), -1).shape[1]
        else:
            input_size = env.observation_space.shape[0]
        
        
        # Add dense layers
        for i, (units, activation, _) in enumerate(self.layer_config):
            self.dense_layers[f'actor_dense_{i}'] = nn.Linear(input_size, units)
            
            if activation == 'relu':
                self.dense_layers[f'actor_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.dense_layers[f'actor_activation_{i}'] = nn.Tanh()
            
            # update input size to be output size of previous layer
            input_size = units
        
        # Initialize weights
        self._init_weights(self.dense_layers, self.layer_config, 'actor')

        # add output layer to dict
        self.dense_layers['actor_output'] = nn.Linear(input_size, env.action_space.shape[0])
        self.dense_layers['actor_activation'] = nn.Tanh()

         # output layer initialization dependent on presence of cnn model  
        if self.cnn_model:
            nn.init.uniform_(self.dense_layers['actor_output'].weight, -3e-4, 3e-4)
            nn.init.uniform_(self.dense_layers['actor_output'].bias, -3e-4, 3e-4)
        else:
            nn.init.uniform_(self.dense_layers['actor_output'].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.dense_layers['actor_output'].bias, -3e-3, 3e-3)

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        # Move the model to the specified device
        self.to(self.device) 


    def forward(self, x):
        x = x.to(self.device)
        if self.cnn_model:
            x = self.cnn_model(x)

        for layer in self.dense_layers.values():
            x = layer(x)
        # x = self.output_layer(x)
        output = x * torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
        
        return output


    def get_config(self):
        config = {
            'env_spec_id': self.env.spec.id if hasattr(self.env, 'spec') else 'Custom/UnknownEnv',
            'dense_layers': self.layer_config,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.learning_rate,
        }

        return config


    def get_clone(self, weights=True):
        # Reconstruct the model from its configuration
        cloned_model = ActorModel(
            env=self.env,
            cnn_model=self.cnn_model,
            dense_layers=self.layer_config,
            optimizer=self.optimizer.__class__.__name__,
            learning_rate=self.learning_rate
        )
        
        if weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())

        return cloned_model


    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "policy_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')

        obj_config = {
            "env_name": self.env.spec.id,
            "dense_layers": self.layer_config,
            "optimizer": self.optimizer_class,
            "learning_rate": self.learning_rate,
        }

        with open(model_dir / "obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)


    @classmethod
    def load(cls, folder):
        model_dir = Path(folder) / "policy_model"
        obj_config_path = model_dir / "obj_config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if obj_config_path.is_file():
            with open(obj_config_path, "r", encoding="utf-8") as f:
                obj_config = json.load(f)
            env = obj_config.get("env_name", "Custom/UnknownEnv")
            cnn_model = obj_config.get("cnn_model", None)
            dense_layers = obj_config.get("dense_layers", [])
            optimizer = obj_config.get("optimizer", "Adam")
            learning_rate = obj_config.get("learning_rate", 0.0001)
        else:
            raise FileNotFoundError(f"No configuration file found in {obj_config_path}")

        actor_model = cls(env, cnn_model, dense_layers, optimizer, learning_rate)
        actor_model.load_state_dict(torch.load(model_path))

        return actor_model


class CriticModel(Model):
    def __init__(self, env, cnn_model: nn.ModuleList=None, state_layers=None, merged_layers=None, optimizer: str = 'Adam', learning_rate=0.001):
        super().__init__()
        self.env = env
        self.cnn_model = cnn_model
        self.state_config = state_layers
        self.merged_config = merged_layers
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        # instantiate ModuleDicts for state and merged Modules
        self.state_layers = nn.ModuleDict()
        self.merged_layers = nn.ModuleDict()

        if self.cnn_model:
            obs_shape = env.observation_space.shape
            dummy_input = torch.zeros(1, *obs_shape, device=self.device)
            dummy_input = dummy_input.permute(0, 3, 1, 2)
            cnn_output = self.cnn_model(dummy_input)
            input_size = cnn_output.view(cnn_output.size(0), -1).shape[1]
        else:
            input_size = env.observation_space.shape[0]

        # Define state processing layers
        for i, (units, activation, _) in enumerate(self.state_config):
            self.state_layers[f'critic_state_dense_{i}'] = nn.Linear(input_size, units)
            if activation == 'relu':
                self.state_layers[f'critic_state_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.state_layers[f'critic_state_activation_{i}'] = nn.Tanh()
            input_size = units

        # Define the action input layer
        action_input_size = np.prod(self.env.action_space.shape)

        # Define merged layers
        for i, (units, activation, _) in enumerate(self.merged_config):
            self.merged_layers[f'critic_merged_dense_{i}'] = nn.Linear(input_size + action_input_size, units)
            if activation == 'relu':
                self.merged_layers[f'critic_merged_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.merged_layers[f'critic_merged_activation_{i}'] = nn.Tanh()
            input_size = units

        # Initialize state and merged layers' weights
        self._init_weights(self.state_layers, self.state_config, 'critic_state')
        self._init_weights(self.merged_layers, self.merged_config, 'critic_merged')

        # add output layer to merged layers
        self.merged_layers['critic_output'] = nn.Linear(input_size, 1)

        # Output layer kernel initialization dependent on presence of cnn model
        if self.cnn_model:
            nn.init.uniform_(self.merged_layers['critic_output'].weight, -3e-4, 3e-4)
            nn.init.uniform_(self.merged_layers['critic_output'].bias, -3e-4, 3e-4)
        else:
            nn.init.uniform_(self.merged_layers['critic_output'].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.merged_layers['critic_output'].bias, -3e-3, 3e-3)
        
        # Define the optimizer
        self.optimizer = self._init_optimizer()

         # Move the model to the specified device
        self.to(self.device)

    # def _init_weights(self):
    #     for i, (_, _, init_config) in enumerate(self.layer_config):
    #         layer = self.dense_layers[f'critic_dense_{i}']
    #         # for (name, layer) in self.dense_layers.items():

    #         ##DEBUG
    #         print(f'layer critic_dense_{i} using {init_config} for {layer}')

    #         if init_config == 'variance scaling':
    #             #DEBUG
    #             torch_utils.VarianceScaling_(layer.weight, **init_config['variance scaling'])
    #             # torch_utils.VarianceScaling_(layer.bias, **init_config['variance scaling'])
            
    #         if init_config == 'uniform':
    #             nn.init.uniform_(layer.weight, **init_config['uniform'])
    #             nn.init.uniform_(layer.bias, **init_config['uniform'])

    #         if init_config == 'normal':
    #             nn.init.normal_(layer.weight, **init_config['normal'])
    #             nn.init.normal_(layer.bias, **init_config['normal'])

    #         if init_config == 'constant':
    #             nn.init.constant_(layer.weight, **init_config['constant'])
    #             nn.init.constant_(layer.bias, **init_config['constant'])

    #         if init_config == 'ones':
    #             nn.init.ones_(layer.weight)
    #             nn.init.ones_(layer.bias)

    #         if init_config == 'zeros':
    #             nn.init.zeros_(layer.weight)
    #             nn.init.zeros_(layer.bias)

    #         if init_config == 'xavier uniform':
    #             nn.init.xavier_uniform_(layer.weight, **init_config['xavier uniform'])
    #             # nn.init.uniform_(layer.bias)

    #         if init_config == 'xavier normal':
    #             nn.init.xavier_normal_(layer.weight, **init_config['xavier normal'])
    #             # nn.init.normal_(layer.bias)

    #         if init_config == 'kaiming uniform':
    #             nn.init.kaiming_uniform_(layer.weight, **init_config['kaiming uniform'])
    #             # nn.init.uniform_(layer.bias)

    #         if init_config == 'kaiming normal':
    #             nn.init.kaiming_normal_(layer.weight, **init_config['kaiming normal'])
    #             # nn.init.normal_(layer.bias)

    #         if init_config == 'truncated normal':
    #             nn.init.trunc_normal_(layer.weight, **init_config['truncated normal'])
    #             nn.init.trunc_normal_(layer.bias, **init_config['truncated normal'])

    # def _init_optimizer(self):
    #     if self.optimizer_class == 'Adam':
    #         return optim.Adam(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class =='SGD':
    #         return optim.SGD(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'RMSprop':
    #         return optim.RMSprop(self.parameters(), lr=self.learning_rate)
    #     elif self.optimizer_class == 'Adagrad':
    #         return optim.Adagrad(self.parameters(), lr=self.learning_rate)
    #     else:
    #         raise NotImplementedError

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        if self.cnn_model:
            state = self.cnn_model(state)
        else:
            state = state.view(state.size(0), -1)  # Flatten state input if not using a cnn_model
        for layer in self.state_layers.values():
            state = layer(state)
        merged = torch.cat([state, action], dim=1)
        for layer in self.merged_layers.values():
            merged = layer(merged)
        return merged
    
    # def _set_learning_rate(self, learning_rate):
    #     """Sets learning rate of optimizer."""
    #     self.optimizer.learning_rate = learning_rate

    def get_config(self):
        config = {
            'env_spec_id': self.env.spec.id if hasattr(self.env, 'spec') and hasattr(self.env.spec, 'id') else 'Custom/UnknownEnv',
            'hidden_layers': len(self.state_layers) + len(self.merged_layers),
            'state_layers': self.state_config,
            'merged_layers': self.merged_config,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer.__class__.__name__,
        }

        return config
    
    def get_clone(self, weights=True):
        # Reconstruct the model from its configuration
        cloned_model = CriticModel(
            env=self.env,
            cnn_model=self.cnn_model,
            state_layers=self.state_config,
            merged_layers=self.merged_config,
            learning_rate=self.learning_rate
        )
        
        if weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())
            
            # Optionally, clone the optimizer (requires more manual work, shown below)
            cloned_optimizer = type(self.optimizer)(cloned_model.parameters(), **self.optimizer.defaults)
            cloned_optimizer.load_state_dict(self.optimizer.state_dict())

        return cloned_model


    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "value_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')

        obj_config = {
            "env_name": self.env.spec.id,
            "state_layers": self.state_config,
            "merged_layers": self.merged_config,
            "learning_rate": self.learning_rate,
        }

        with open(model_dir / "obj_config.json", "w", encoding="utf-8") as f:
            json.dump(obj_config, f)


    @classmethod
    def load(cls, folder):
        model_dir = Path(folder) / "value_model"
        obj_config_path = model_dir / "obj_config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if obj_config_path.is_file():
            with open(obj_config_path, "r", encoding="utf-8") as f:
                obj_config = json.load(f)
            env = obj_config.get("env_name", "Custom/UnknownEnv")
            cnn_model = obj_config.get("cnn_model", None)
            dense_layers = obj_config.get("dense_layers", [])
            optimizer = obj_config.get("optimizer", "Adam")
            learning_rate = obj_config.get("learning_rate", 0.0001)
        else:
            raise FileNotFoundError(f"No configuration file found in {obj_config_path}")

        model = cls(env, cnn_model, dense_layers, optimizer, learning_rate)
        model.load_state_dict(torch.load(model_path))

        return model


def build_layers(units_per_layer: List[int], activation: str, initializer: str):
    """formats sweep_config into policy and value layers"""
    # get policy layers
    layers = []
    for units in units_per_layer:
        layers.append((units, activation, initializer))
    return layers
