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

    def _init_weights(self, module_dict, layer_config):
        config_index = 0

        for layer_name, layer in module_dict.items():
            if 'dense' in layer_name:
                if isinstance(layer_config, list):
                    _, _, init_config = layer_config[config_index]
                elif isinstance(layer_config, str):
                    init_config=layer_config

                ##DEBUG
                # print(f'layer {prefix}_dense_{config_index} using {init_config} for {layer}')
                
                if isinstance(init_config, dict):
                    if 'default' in init_config:
                        pass

                    if 'variance scaling' in init_config:
                        # print('dict variance scaling')
                        torch_utils.VarianceScaling_(layer.weight, **init_config['variance scaling'])
                    
                    elif 'xavier uniform' in init_config:
                        nn.init.xavier_uniform_(layer.weight, **init_config['xavier uniform'])

                    elif 'xavier normal' in init_config:
                        nn.init.xavier_normal_(layer.weight, **init_config['xavier normal'])

                    elif 'kaiming uniform' in init_config:
                        nn.init.kaiming_uniform_(layer.weight, **init_config['kaiming uniform'])
                    
                    elif 'kaiming normal' in init_config:
                        nn.init.kaiming_normal_(layer.weight, **init_config['kaiming normal'])

                    elif 'truncated normal' in init_config:
                        nn.init.trunc_normal_(layer.weight, **init_config['truncated normal'])
                        nn.init.trunc_normal_(layer.bias, **init_config['truncated normal'])

                    elif 'uniform' in init_config:
                        nn.init.uniform_(layer.weight, **init_config['uniform'])
                        nn.init.uniform_(layer.bias, **init_config['uniform'])

                    elif 'normal' in init_config:
                        nn.init.normal_(layer.weight, **init_config['normal'])
                        nn.init.normal_(layer.bias, **init_config['normal'])

                    elif 'constant' in init_config:
                        nn.init.constant_(layer.weight, **init_config['constant'])
                        nn.init.constant_(layer.bias, **init_config['constant'])
                
                elif isinstance(init_config, str):
                    if init_config == 'default':
                        pass

                    if init_config == 'variance scaling':
                        torch_utils.VarianceScaling_(layer.weight)

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
                
                else:
                    raise ValueError(f"Invalid init_config {init_config} index {config_index}")
                
                config_index += 1


    def _init_optimizer(self):
        if self.optimizer_class == 'Adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_class =='SGD':
            return optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_class == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_class == 'Adagrad':
            return optim.Adagrad(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
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
        optimizer_params:dict={},
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.env = env
        self.layer_config = dense_layers
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
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
            'num_layers': len(self.dense_layers),
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
        torch.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, config_path, load_weights=True):
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = config.get("env")
            dense_layers = config.get("dense_layers")
            optimizer = config.get("optimizer")
            learning_rate = config.get("learning_rate")
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        model = cls(env, dense_layers, optimizer, learning_rate)

        # Load weights if True
        if load_weights:
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
        optimizer_params:dict={},
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.env = env
        self.layer_config = dense_layers
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
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

        config = self.get_config()

        return config


    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "value_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')

        config = {
            "env": self.env.spec.id,
            "dense_layers": self.layer_config,
            "optimizer": self.optimizer_class,
            "learning_rate": self.learning_rate,
        }

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights:bool=True):
        model_dir = Path(config_path) / "value_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = config.get("env")
            dense_layers = config.get("dense_layers")
            optimizer = config.get("optimizer")
            learning_rate = config.get("learning_rate")
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        model = cls(env, dense_layers, optimizer, learning_rate)

        # Load weights if True
        if load_weights:
            model.load_state_dict(torch.load(model_path))

        return model


class ActorModel(Model):
    
    def __init__(self, env, cnn_model=None, dense_layers=None, output_layer_kernel=None, goal_shape:tuple=None, optimizer: str = 'Adam',
                 optimizer_params:dict={}, learning_rate=0.0001, normalize_layers:bool=False,
                 clamp_output:float=None, device=None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.layer_config = dense_layers
        self.output_config = output_layer_kernel
        self.cnn_model = cnn_model
        self.goal_shape = goal_shape
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.learning_rate = learning_rate
        self.normalize_layers = normalize_layers
        self.clamp_output = clamp_output

        # set internal attributes
        # get observation space
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            self._obs_space_shape = env.observation_space['observation'].shape
        else:
            self._obs_space_shape = env.observation_space.shape

        self.dense_layers = nn.ModuleDict()
        self.output_layer = nn.ModuleDict()
        self.output_activation = nn.ModuleDict()
        
        # if self.cnn_model:
        #     obs_shape = env.observation_space.shape
        #     dummy_input = torch.zeros(1, *obs_shape, device=self.device)
        #     dummy_input = dummy_input.permute(0, 3, 1, 2)
        #     cnn_output = self.cnn_model(dummy_input)
        #     input_size = cnn_output.view(cnn_output.size(0), -1).shape[1]
        # else:
        #     input_size = env.observation_space.shape[0]

        # Adding support for goal
        if self.cnn_model:
            # obs_shape = env.observation_space.shape
            dummy_input = torch.zeros(1, *self._obs_space_shape, device=self.device)
            dummy_input = dummy_input.permute(0, 3, 1, 2)
            cnn_output = self.cnn_model(dummy_input)
            cnn_output_size = cnn_output.size(1)
            
            if self.goal_shape is not None:
                input_size = cnn_output_size + self.goal_shape[0]
            else:
                input_size = cnn_output_size
        else:
            input_size = self._obs_space_shape[0]
            
            if self.goal_shape is not None:
                input_size += self.goal_shape[0]
        
        # DEBUG
        print(f'actor layer config: {self.layer_config}')
        print(f'actor input size: {input_size}')
        
        # Add dense layers
        for i, (units, activation, _) in enumerate(self.layer_config):
            self.dense_layers[f'actor_dense_{i}'] = nn.Linear(input_size, units)
            
            # add normalization layer if normalize
            if self.normalize_layers:
                self.dense_layers[f'actor_normalization_{i}'] = nn.LayerNorm(units)
            
            # add activation layer
            if activation == 'relu':
                self.dense_layers[f'actor_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.dense_layers[f'actor_activation_{i}'] = nn.Tanh()
            
            # update input size to be output size of previous layer
            input_size = units
        
        # Initialize weights
        self._init_weights(self.dense_layers, self.layer_config)

        # add output layer to dict
        self.output_layer['actor_dense_output'] = nn.Linear(input_size, env.action_space.shape[0])
        self.output_activation['actor_output_activation'] = nn.Tanh()

        # output layer initialization dependent on presence of cnn model  
        # if self.cnn_model:
        #     nn.init.uniform_(self.output_layer['actor_output'].weight, -3e-4, 3e-4)
        #     nn.init.uniform_(self.output_layer['actor_output'].bias, -3e-4, 3e-4)
        # else:
        #     nn.init.uniform_(self.output_layer['actor_output'].weight, -3e-3, 3e-3)
        #     nn.init.uniform_(self.output_layer['actor_output'].bias, -3e-3, 3e-3)

        # UPDATE output layer kernel initializer to be hyperparam
        self._init_weights(self.output_layer, self.output_config)

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        # Move the model to the specified device
        self.to(self.device) 


    def forward(self, x, goal=None):
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)

        if self.cnn_model:
            x = self.cnn_model(x)

        if self.goal_shape is not None:
            x = torch.cat([x, goal], dim=-1)

        for layer in self.dense_layers.values():
            x = layer(x)

        for layer in self.output_layer.values():
            mu = layer(x)
        
        for layer in self.output_activation.values():
            pi = layer(mu)
        
        if self.clamp_output is not None:
            # print('clamp output fired...')
            pi = torch.clamp(pi * torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device), -self.clamp_output, self.clamp_output)
            # print(f'pi: {pi}')
            return mu, pi
        
        # print('unclamped output fired')
        pi = pi * torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
        # print(f'pi: {pi}')
        return mu, pi

    def get_config(self):
        config = {
            'env': self.env.spec.id if hasattr(self.env, 'spec') else 'Custom/UnknownEnv',
            'cnn_model': self.cnn_model.get_config() if self.cnn_model is not None else None,
            'num_layers': len(self.dense_layers),
            'dense_layers': self.layer_config,
            'output_layer_kernel':self.output_config,
            'goal_shape': self.goal_shape,
            'optimizer': self.optimizer.__class__.__name__,
            'optimizer_params': self.optimizer_params,
            'learning_rate': self.learning_rate,
            'normalize_layers': self.normalize_layers,
            'clamp_output': self.clamp_output,
        }

        return config


    def get_clone(self, weights=True):
        # Reconstruct the model from its configuration
        cloned_model = ActorModel(
            env=self.env,
            cnn_model=self.cnn_model,
            dense_layers=self.layer_config,
            output_layer_kernel=self.output_config,
            goal_shape=self.goal_shape,
            optimizer=self.optimizer_class,
            optimizer_params=self.optimizer_params,
            learning_rate=self.learning_rate,
            normalize_layers=self.normalize_layers,
            clamp_output=self.clamp_output,
            device=self.device
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
        torch.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights=True):
        model_dir = Path(config_path) + "policy_model"
        config_path = model_dir + "config.json"
        model_path = model_dir + 'pytorch_model.pt'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = gym.make(config.get("env"))
            cnn_model = config.get("cnn_model", None)
            if cnn_model:
                cnn_model = cnn_models.CNN(cnn_model['layers'], env)
            dense_layers = config.get("dense_layers")
            goal_shape = config.get("goal_shape", None)
            optimizer = config.get("optimizer")
            optimizer_params = config.get("optimizer_params")
            learning_rate = config.get("learning_rate")
            normalize = config.get("normalize", False)
            clamp_output = config.get("clamp_output", None)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        actor_model = cls(env, cnn_model, dense_layers, goal_shape, optimizer, optimizer_params, learning_rate, normalize, clamp_output)
        
        # Load weights if True
        if load_weights:
            actor_model.load_state_dict(torch.load(model_path))

        return actor_model


class CriticModel(Model):
    def __init__(self, env, cnn_model=None, state_layers=None, merged_layers=None, output_layer_kernel=None,
                 goal_shape:tuple=None, optimizer: str = 'Adam', optimizer_params:dict={},
                 learning_rate=0.001, normalize_layers:bool=False, device=None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.cnn_model = cnn_model
        self.state_config = state_layers
        self.merged_config = merged_layers
        self.output_config = output_layer_kernel
        self.goal_shape = goal_shape
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.learning_rate = learning_rate
        self.normalize_layers = normalize_layers

        # set internal attributes
        # get observation space
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            self._obs_space_shape = env.observation_space['observation'].shape
        else:
            self._obs_space_shape = env.observation_space.shape

        # instantiate ModuleDicts for state and merged Modules
        self.state_layers = nn.ModuleDict()
        self.merged_layers = nn.ModuleDict()
        self.output_layer = nn.ModuleDict()

        # Adding support for goal
        if self.cnn_model:
            # obs_shape = env.observation_space.shape
            dummy_input = torch.zeros(1, *self._obs_space_shape, device=self.device)
            dummy_input = dummy_input.permute(0, 3, 1, 2)
            cnn_output = self.cnn_model(dummy_input)
            cnn_output_size = cnn_output.size(1)
            
            if self.goal_shape is not None:
                input_size = cnn_output_size + self.goal_shape[0]
            else:
                input_size = cnn_output_size
        else:
            input_size = self._obs_space_shape[0]
            
            if self.goal_shape is not None:
                input_size += self.goal_shape[0]

        # Define state processing layers
        for i, (units, activation, _) in enumerate(self.state_config):
            self.state_layers[f'critic_state_dense_{i}'] = nn.Linear(input_size, units)

            # add normalization layer if normalize
            if self.normalize_layers:
                self.state_layers[f'critic_state_normalize_{i}'] = nn.LayerNorm(units)

            # add activation layer
            if activation == 'relu':
                self.state_layers[f'critic_state_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.state_layers[f'critic_state_activation_{i}'] = nn.Tanh()
            input_size = units

        # Define the action input layer
        action_input_size = np.prod(self.env.action_space.shape)

        # Define merged layers
        for i, (units, activation, _) in enumerate(self.merged_config):
            if i == 0:
                # For the first merged layer, concatenate state (and goal, if present) with action
                self.merged_layers[f'critic_merged_dense_{i}'] = nn.Linear(input_size + action_input_size, units)
            else:
                # For subsequent merged layers, use the output size of the previous layer as input size
                self.merged_layers[f'critic_merged_dense_{i}'] = nn.Linear(input_size, units)
            # add normalization layer if normalize
            if self.normalize_layers:
                self.merged_layers[f'critic_merged_normalize_{i}'] = nn.LayerNorm(units)
            # add activation layer
            if activation == 'relu':
                self.merged_layers[f'critic_merged_activation_{i}'] = nn.ReLU()
            elif activation == 'tanh':
                self.merged_layers[f'critic_merged_activation_{i}'] = nn.Tanh()
            input_size = units

        # Initialize state and merged layers' weights
        self._init_weights(self.state_layers, self.state_config)
        self._init_weights(self.merged_layers, self.merged_config)

        # add output layer to merged layers
        self.output_layer['critic_dense_output'] = nn.Linear(input_size, 1)

        # Output layer kernel initialization dependent on presence of cnn model
        # if self.cnn_model:
        #     nn.init.uniform_(self.output_layer['critic_output'].weight, -3e-4, 3e-4)
        #     nn.init.uniform_(self.output_layer['critic_output'].bias, -3e-4, 3e-4)
        # else:
        #     nn.init.uniform_(self.output_layer['critic_output'].weight, -3e-3, 3e-3)
        #     nn.init.uniform_(self.output_layer['critic_output'].bias, -3e-3, 3e-3)
        
        # UPDATE output layer kernel to be hyperparam
        self._init_weights(self.output_layer, self.output_config)

        # Define the optimizer
        self.optimizer = self._init_optimizer()

         # Move the model to the specified device
        self.to(self.device)

    def forward(self, state, action, goal=None):
        state = state.to(self.device)
        action = action.to(self.device)
        if goal is not None:
            goal = goal.to(self.device)

        if self.cnn_model:
            state = self.cnn_model(state)
        # else: cnn model already flattens output
        #     state = state.view(state.size(0), -1)  # Flatten state input if not using a cnn_model

        if self.goal_shape is not None:
            state = torch.cat([state, goal], dim=-1)

        for layer in self.state_layers.values():
            state = layer(state)

        merged = torch.cat([state, action], dim=-1)
        for layer in self.merged_layers.values():
            merged = layer(merged)

        for layer in self.output_layer.values():
            output = layer(merged)
        
        return output
    
    # def _set_learning_rate(self, learning_rate):
    #     """Sets learning rate of optimizer."""
    #     self.optimizer.learning_rate = learning_rate

    def get_config(self):
        config = {
            'env': self.env.spec.id if hasattr(self.env, 'spec') and hasattr(self.env.spec, 'id') else 'Custom/UnknownEnv',
            'cnn_model': self.cnn_model.get_config() if self.cnn_model is not None else None,
            'num_layers': len(self.state_layers) + len(self.merged_layers),
            'state_layers': self.state_config,
            'merged_layers': self.merged_config,
            'output_layer_kernel': self.output_config,
            'goal_shape': self.goal_shape,
            'optimizer': self.optimizer.__class__.__name__,
            'optimizer_params': self.optimizer_params,
            'learning_rate': self.learning_rate,
            'normalize_layers': self.normalize_layers,
        }

        return config
    
    def get_clone(self, weights=True):
        # Reconstruct the model from its configuration
        cloned_model = CriticModel(
            env=self.env,
            cnn_model=self.cnn_model,
            state_layers=self.state_config,
            merged_layers=self.merged_config,
            output_layer_kernel=self.output_config,
            goal_shape=self.goal_shape,
            optimizer=self.optimizer_class,
            optimizer_params=self.optimizer_params,
            learning_rate=self.learning_rate,
            normalize_layers=self.normalize_layers,
            device=self.device
        )
        
        if weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())
            
            # # Optionally, clone the optimizer (requires more manual work, shown below)
            # cloned_optimizer = type(self.optimizer)(cloned_model.parameters(), **self.optimizer.defaults)
            # cloned_optimizer.load_state_dict(self.optimizer.state_dict())

        return cloned_model


    def save(self, folder):
        # Ensure the model directory exists
        model_dir = Path(folder) / "value_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        torch.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        torch.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights=True):
        model_dir = Path(config_path) + "value_model"
        config_path = model_dir + "config.json"
        model_path = model_dir + 'pytorch_model.pt'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = gym.make(config.get("env"))
            cnn_model = config.get("cnn_model", None)
            if cnn_model:
                cnn_model = cnn_models.CNN(cnn_model['layers'], env)
            state_layers = config.get("state_layers")
            merged_layers = config.get("merged_layers")
            goal_shape = config.get("goal_shape", None)
            optimizer = config.get("optimizer")
            learning_rate = config.get("learning_rate")
            optimizer_params = config.get("optimizer_params")
            normalize = config.get("normalize", False)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        model = cls(env, cnn_model, state_layers, merged_layers, goal_shape, optimizer, optimizer_params, learning_rate, normalize)
        
        # Load weights if True
        if load_weights:
            model.load_state_dict(torch.load(model_path))

        return model


def build_layers(units_per_layer: List[int], activation: str, initializer: str):
    """formats config into policy and value layers"""
    # get policy layers
    layers = []
    for units in units_per_layer:
        layers.append((units, activation, initializer))
    return layers
