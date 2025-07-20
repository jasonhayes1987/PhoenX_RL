"""Holds Model classes used for Reinforcement learning."""

# imports
from abc import abstractmethod
import json
import os
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import torch as T
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical, Beta, Normal
# from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from torch_utils import get_device, VarianceScaling_
# from logging_config import logger
from env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from utils import check_for_inf_or_NaN
from schedulers import ScheduleWrapper

class Model(nn.Module):
    """
    Base class for all reinforcement learning models.

    This class dynamically constructs a neural network based on the provided layer configuration
    and supports various optimizers and learning rate schedulers.

    Attributes:
        env (EnvWrapper): The environment wrapper for the model.
        layer_config (list): List of dictionaries specifying the layers and their parameters.
        optimizer_params (dict): Dictionary specifying optimizer type and parameters.
        scheduler_params (dict): Dictionary specifying scheduler type and parameters (optional).
        device (str): The device ('cpu' or 'cuda') to run the model on.
    """
    def __init__(self, env: EnvWrapper, layer_config, optimizer_params: dict = None,
                 lr_scheduler: ScheduleWrapper = None, device=None):
        """
        Sets up the module dictionary of layers (most of which
        will be lazy).

        Args:
            env (EnvWrapper): Environment wrapper.
            layer_config (list): List of dictionaries specifying the layers and params.
            optimizer_params (dict): Optimizer configuration.
            scheduler_params (dict): LR scheduler configuration.
            device (str): Device to run on.
        """
        super().__init__()
        self.env = env
        self.layer_config = layer_config
        self.layers = nn.ModuleDict()
        self.optimizer_params = optimizer_params or {'type': 'Adam', 'params': {'lr': 0.001}}
        self.lr_scheduler = lr_scheduler
        self.device = get_device(device)

        # Build the layers dynamically based on config
        for i, layer_info in enumerate(self.layer_config):
            layer_type = layer_info['type']
            layer_params = layer_info.get('params', {})
            self.layers[f'{layer_type}_{i}'] = self._build_layer(layer_type, layer_params)

        # Set optimizer to None (set in init_parameters function after dry run)
        self.optimizer = None

        # Move the model to device
        self.to(self.device)
        
    def _init_model(self, module_dict: nn.ModuleDict, layer_config: list):
        """
        Performs a "dry run" forward pass with dummy_input to initialize
        all lazy modules. Then, initializes weights and optimizer/scheduler.

        Args:
            dummy_input (Tensor, optional): If None, automatically creates
                a dummy input based on env.observation_space.shape. If your
                environment is a 3D image (C, H, W), use (1, C, H, W).
        """
        obs_space = (self.env.single_observation_space if hasattr(self.env, "single_observation_space") 
                        else self.env.observation_space)
        # Dry run forward pass to initialize lazy modules
        # Check if the observation space is a dictionary for goal-aware environments
        if isinstance(obs_space, gym.spaces.Dict):
            obs_shape = obs_space['observation'].shape
            goal_shape = obs_space['desired_goal'].shape
            state_input = T.ones((32, *obs_shape), device=self.device, dtype=T.float)
            goal_input = T.ones((32, *goal_shape), device=self.device, dtype=T.float)
            # Check if CriticModel instance to pass action dummy values
            if isinstance(self, CriticModel):
                action_shape = self.env.single_action_space.shape
                action_input = T.ones((32, *action_shape), device=self.device, dtype=T.float)
                with T.no_grad():
                    _ = self.forward(state_input, action_input, goal_input)
            else:
                with T.no_grad():
                    _ = self.forward(state_input, goal_input)
        else:
            obs_shape = obs_space.shape
            #DEBUG
            # print(f'init model obs_shape:{obs_shape}')
            state_input = T.ones((32, *obs_shape), device=self.device, dtype=T.float)
            #DEBUG
            # print(f'state input shape:{state_input.shape}')
            if isinstance(self, CriticModel):
                action_shape = self.env.single_action_space.shape
                action_input = T.ones((32, *action_shape), device=self.device, dtype=T.float)
                with T.no_grad():
                    _ = self.forward(state_input, action_input)
            else:
                with T.no_grad():
                    _ = self.forward(state_input)

        # Initialize weights after lazy modules are materialized
        self._init_weights(layer_config, module_dict)

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer()

    def _build_layer(self, layer_type, params):
        """
        Build a specific layer based on its type and parameters.

        Args:
            layer_type (str): Type of the layer (e.g., 'dense', 'conv2d', etc.).
            params (dict): Parameters for the layer.

        Returns:
            nn.Module: Constructed layer.
        """
        if layer_type == 'dense':
            return nn.LazyLinear(params["units"])

        elif layer_type == 'conv2d':
            return nn.LazyConv2d(
                out_channels=params.get('out_channels', 64),
                kernel_size=params.get('kernel_size', 3),
                stride=params.get('stride', 1),
                padding=params.get('padding', 0),
                bias=True
            )

        elif layer_type == 'pool':
            return nn.MaxPool2d(**params)

        elif layer_type == 'dropout':
            return nn.Dropout(**params)

        elif layer_type == 'batchnorm1d':
            return nn.LazyBatchNorm1d()

        elif layer_type == 'batchnorm2d':
            return nn.LazyBatchNorm2d()

        elif layer_type == 'layernorm':
            return nn.LayerNorm(**params)

        elif layer_type == 'flatten':
            return nn.Flatten()

        elif layer_type == 'relu':
            return nn.ReLU()
        
        elif layer_type == 'leakyrelu':
            return nn.LeakyReLU()

        elif layer_type == 'tanh':
            return nn.Tanh()

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def _init_weights(self, layer_config, layers):
        """
        Initialize the weights for the model.

        Args:
            layer_config (dict): configuration of layer.
            layers (torch layers): torch.nn.Module.layers.
        """
        # Loop through each layer config and corresponding layer
        for config, (layer_name, layer) in zip(layer_config, layers.items()):
            if not hasattr(layer, 'weight'):
                continue
            
            # If the params of the layer config dict contains a kernel, apply it to layer
            # if config['type'] in ['dense', 'transformer']:
            kernel = config.get('params', {}).get('kernel', 'default')  # Get kernel or 'default'
            kernel_params = config.get('params', {}).get('kernel params', {}) # Get kernel params or empty dict
            # Apply the specified initialization scheme
            if kernel == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, **kernel_params)
            elif kernel == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight)
            elif kernel == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif kernel == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif kernel == 'truncated_normal':
                nn.init.trunc_normal_(layer.weight, **kernel_params)
                # nn.init.trunc_normal_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'uniform':
                nn.init.uniform_(layer.weight, **kernel_params)
                # nn.init.uniform_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'normal':
                nn.init.normal_(layer.weight, **kernel_params)
                # nn.init.normal_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'orthogonal':
                nn.init.orthogonal_(layer.weight, **kernel_params)
            elif kernel == 'constant':
                nn.init.constant_(layer.weight, **kernel_params)
                # nn.init.constant_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'ones':
                nn.init.ones_(layer.weight, **kernel_params)
                # nn.init.ones_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'zeros':
                nn.init.zeros_(layer.weight, **kernel_params)
                # nn.init.zeros_(layer.bias, **config['params']['kernel params'])
            elif kernel == 'variance_scaling':
                VarianceScaling_(layer.weight, **kernel_params)
            elif kernel == 'default':
                # Use PyTorch's default initialization (skip)
                pass
            else:
                raise ValueError(f"Unsupported initialization: {kernel}")
            # Initialize bias
            if hasattr(layer, 'bias'):
                nn.init.zeros_(layer.bias)

    def _init_optimizer(self):
        """
        Initialize the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer_type = self.optimizer_params['type']
        optimizer_params = self.optimizer_params['params']
        if optimizer_type == 'Adam':
            return optim.Adam(self.parameters(), **optimizer_params)
        elif optimizer_type == 'SGD':
            return optim.SGD(self.parameters(), **optimizer_params)
        elif optimizer_type == 'RMSprop':
            return optim.RMSprop(self.parameters(), **optimizer_params)
        elif optimizer_type == 'Adagrad':
            return optim.Adagrad(self.parameters(), **optimizer_params)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {optimizer_type}")
    
    # def _init_scheduler(self):
    #     """
    #     Initialize the learning rate scheduler for the model.

    #     Args:
    #         scheduler (dict): Scheduler configuration.

    #     Returns:
    #         torch.optim.lr_scheduler: Configured scheduler.
    #     """
    #     scheduler_type = self.scheduler_params.get('type', '').lower()
    #     scheduler_params = self.scheduler_params.get('params', {})
    #     if scheduler_type == 'cosineannealing':
    #         return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
    #     elif scheduler_type == 'step':
    #         return optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
    #     elif scheduler_type == 'exponential':
    #         return optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)
    #     elif scheduler_type == 'linear':
    #         return optim.lr_scheduler.LinearLR(self.optimizer, **scheduler_params)
    #     else:
    #         raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        pass

    @abstractmethod
    def get_config(self):
        """
        Retrieve the model configuration.

        Returns:
            dict: Configuration dictionary.
        """
        pass

    @abstractmethod
    def save(self, folder):
        """
        Save the model and its configuration.

        Args:
            folder (str): Directory to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, folder):
        """
        Load a model from a saved configuration.

        Args:
            folder (str): Directory containing the saved model.

        Returns:
            Model: Loaded model instance.
        """
        pass


class StochasticDiscretePolicy(Model):
    """
    Policy model for predicting a probability distribution over a discrete action space.

    This class builds on the `Model` base class and adds functionality specific to
    policies with a discrete action space, such as using a Categorical distribution
    for action selection.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): Configuration of hidden layers.
        output_layer_kernel (dict): Configuration of the output layer weights.
        optimizer_params (dict): Parameters for the optimizer. (default: Adam with lr=0.001)
        scheduler_params (dict): Parameters for the learning rate scheduler (optional).
        distribution (str): Type of distribution for action selection (default: 'categorical').
        device (str): Device to run the model on (default: 'cuda').
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: list,
        output_layer_kernel: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params:dict = None,
        distribution: str = 'categorical',
        device: str = None
    ):
        """
        Initialize the policy model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list): List of dictionaries specifying hidden layer configurations.
            output_layer_kernel (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            scheduler_params (dict, optional): Scheduler parameters (default: None).
            distribution (str): Type of distribution for actions (default: 'categorical').
            device (str): Device for computation (default: 'cuda').
        """
        
        super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel
        self.distribution = distribution

        # Get the action space of the environment
        # self.act_space = self.env.single_action_space if isinstance(self.env, GymnasiumWrapper) else self.env.action_space
        self.act_space = (self.env.single_action_space 
                          if hasattr(self.env, "single_action_space") 
                          else self.env.action_space)
        num_actions = self.act_space.n

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'policy_dense_output': nn.LazyLinear(num_actions)
        })
        # self.add_module('output_layer', self.output_layer)

        # Initialize weights
        # self._init_weights(self.layer_config, self.layers)
        # self._init_weights(self.output_config, self.output_layer)

        # Initialize optimizer
        # self.optimizer = self._init_optimizer()

        # Move to device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).

        Returns:
            Tuple[Categorical, Tensor]: Action distribution and logits for the action space.
        """
        #DEBUG
        # print(f'discrete policy shape of x: {x.shape}')
        # print(f'discrete policy x:{x}')
        if x.dim() == 1: # Check if tensor is flat
            x = x.unsqueeze(-1)  # Reshape to (batch, 1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Check if observation is image-like (HWC)
        if isinstance(self.env, GymnasiumWrapper):
            # obs_shape = self.env.single_observation_space.shape
            #DEBUG
            # print(f'observation space shape:{obs_shape}')
            if x.dim() == 4 and x.shape[-1] in [3,4]:
                # DEBUG
                # print(f'permutation fired')
                x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)
        #DEBUG
        # print(f'discrete policy new x shape:{x.shape}')
        x = x.to(self.device)
        for layer in self.layers.values():
            x = layer(x)
        x = self.output_layer['policy_dense_output'](x)
        if self.distribution == 'categorical':
            dist = Categorical(logits=x)
            return dist, x
        else:
            raise ValueError(f'Distribution {self.distribution} not supported.')

    def get_config(self):
        """
        Retrieve the configuration of the policy model.

        Returns:
            dict: Configuration dictionary with details about the model.
        """
        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'distribution': self.distribution,
            'device': self.device.type,
        }
        return config

    def save(self, folder):
        """
        Save the model to the specified folder.

        Args:
            folder (str): Path to the directory where the model should be saved.
        """
        # Ensure the model directory exists
        model_dir = Path(folder) / "policy_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        # Save the model configuration
        config = self.get_config()
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, config_path, load_weights=True):
        """
        Load a policy model from a saved configuration.

        Args:
            config_path (str): Path to the configuration file.
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            StochasticDiscretePolicy: Loaded policy model instance.
        """
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # # Determine which wrapper to use
        # wrapper_dict = json.loads(config['env'])
        # wrapper_type = wrapper_dict.get("type")
        # if wrapper_type == 'GymnasiumWrapper':
        #     env = GymnasiumWrapper(wrapper_dict.get("env"))
        # else:
        #     raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

        env = EnvWrapper.from_json(config.get("env"))

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_layer_kernel = config.get("output_layer_kernel", {"default":{}}),
                    optimizer_params = config.get("optimizer_params", {}),
                    scheduler_params = config.get("scheduler_params", None),
                    distribution = config.get("distribution", "categorical"),
                    device = config.get("device", "cpu")
                    )

        # Load weights if True
        if load_weights:
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model

class StochasticContinuousPolicy(Model):
    """
    Policy model for predicting a probability distribution over a continuous action space.

    This class extends the `Model` base class to implement policies for continuous action spaces,
    supporting Beta and Normal distributions.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): Configuration of hidden layers.
        output_layer_kernel (dict): Configuration of the output layer weights.
        optimizer_params (dict): Parameters for the optimizer.
        scheduler_params (dict): Parameters for the learning rate scheduler (optional).
        distribution (str): Type of distribution for action selection ('beta' or 'normal').
        device (str): Device to run the model on (default: 'cuda').
    """

    def __init__(
        self,
        env:EnvWrapper,
        layer_config: List[Dict],
        output_layer_kernel: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params:dict = None,
        distribution: str = 'beta',
        device: str = None
    ):
        """
        Initialize the policy model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list): List of dictionaries specifying hidden layer configurations.
            output_layer_kernel (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            scheduler_params (dict, optional): Scheduler parameters (default: None).
            distribution (str): Type of distribution for actions (default: 'beta').
            device (str): Device for computation (default: 'cuda').
        """
        super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel
        self.distribution = distribution
        # Get the action space of the environment
        self.act_space = (self.env.single_action_space 
                          if hasattr(self.env, "single_action_space") 
                          else self.env.action_space)
        
        num_actions = self.act_space.shape[-1]
        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'policy_output_param_1': nn.LazyLinear(num_actions),
            'policy_output_param_2': nn.LazyLinear(num_actions),
        })

        # Move model to device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).

        Returns:
            Distribution, Tensor, Tensor: Action distribution and its parameters.
        """
        #DEBUG
        # print(f'state shape sent to policy forward:{x.shape}')
        if x.dim() == 1: # Check if tensor is flat
            x = x.unsqueeze(-1)  # Reshape to (batch, 1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Check if observation is image-like (HWC)
        if isinstance(self.env, GymnasiumWrapper):
            # obs_shape = self.env.single_observation_space.shape
            #DEBUG
            # print(f'observation space shape:{obs_shape}')
            if x.dim() == 4 and x.shape[-1] in [3,4]:
                # DEBUG
                # print(f'permutation fired')
                x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)
        x = x.to(self.device)
        for layer in self.layers.values():
            x = layer(x)
        param_1 = self.output_layer['policy_output_param_1'](x)
        param_2 = self.output_layer['policy_output_param_2'](x)
        #DEBUG
        # print(f'param 1 shape:{param_1.shape}')
        # print(f'param 2 shape:{param_2.shape}')
        if self.distribution == 'beta':
            alpha = F.softplus(param_1) + 1.0
            beta = F.softplus(param_2) + 1.0
            dist = Beta(alpha, beta)
            return dist, alpha, beta
        elif self.distribution == 'normal':
            mu = param_1
            sigma = F.softplus(param_2)
            dist = Normal(mu, sigma)
            return dist, mu, sigma
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

    def get_config(self):
        """
        Retrieve the configuration of the policy model.

        Returns:
            dict: Configuration dictionary with details about the model.
        """
        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'distribution': self.distribution,
            'device': self.device.type,
        }
        return config

    def save(self, folder):
        """
        Save the model to the specified folder.

        Args:
            folder (str): Path to the directory where the model should be saved.
        """
        model_dir = Path(folder) / "policy_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, config_path, load_weights=True):
        """
        Load a policy model from a saved configuration.

        Args:
            config_path (str): Path to the configuration file.
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            StochasticContinuousPolicy: Loaded policy model instance.
        """
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # Determine which wrapper to use
        # wrapper_dict = json.loads(config['env'])
        # wrapper_type = wrapper_dict.get("type")
        # if wrapper_type == 'GymnasiumWrapper':
        #     env = GymnasiumWrapper(wrapper_dict.get("env"))
        # else:
        #     raise ValueError(f"Unsupported wrapper type: {wrapper_type}")
        #DEBUG
        # print(f'config:{config}')
        env = EnvWrapper.from_json(config.get("env"))
        #DEBUG
        # print(f'env:{env.config}')

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_layer_kernel = config.get("output_layer_kernel", {"default":{}}),
                    optimizer_params = config.get("optimizer_params", {}),
                    scheduler_params = config.get("scheduler_params", None),
                    distribution = config.get("distribution", "beta"),
                    device = config.get("device", "cpu")
                    )

        # Load weights if True
        if load_weights:
            model.load_state_dict(T.load(model_path))

        return model


class ValueModel(Model):
    """
    Value model for predicting state values.

    This class extends the `Model` base class to implement a neural network for value function approximation in reinforcement learning.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): Configuration of hidden layers.
        output_layer_kernel (dict): Configuration of the output layer weights.
        optimizer_params (dict): Parameters for the optimizer.
        scheduler_params (dict): Parameters for the learning rate scheduler (optional).
        device (str): Device to run the model on (default: 'cuda').
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_layer_kernel: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params = None,
        device = None
    ):
        """
        Initialize the value model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list): List of dictionaries specifying hidden layer configurations.
            output_layer_kernel (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            scheduler_params (dict, optional): Scheduler parameters (default: None).
            device (str): Device for computation (default: 'cuda').
        """
        super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'value_dense_output': nn.LazyLinear(1)
        })
        self.add_module('output_layer', self.output_layer)

        # Move model to device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).

        Returns:
            Tensor: Predicted state value.
        """
        #DEBUG
        # print(f'value model x shape:{x.shape}')
        # print(f'value model x:{x}')
        if x.dim() == 1: # Check if tensor is flat
            x = x.unsqueeze(-1)  # Reshape to (batch, 1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Check if observation is image-like (HWC)
        if isinstance(self.env, GymnasiumWrapper):
            obs_shape = self.env.single_observation_space.shape
            #DEBUG
            # print(f'observation space shape:{obs_shape}')
            if x.dim() == 4 and x.shape[-1] in [3,4]:
                # DEBUG
                # print(f'permutation fired')
                x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)
        #DEBUG
        # print(f'value model new x shape:{x.shape}')
        x = x.to(self.device)
        #DEBUG
        # print(f'Value Model Input Shape: {x.size()}')
        for layer in self.layers.values():
            x = layer(x)

        x = self.output_layer['value_dense_output'](x)

        return x


    def get_config(self):
        """
        Retrieve the configuration of the value model.

        Returns:
            dict: Configuration dictionary with details about the model.
        """

        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'device': self.device.type,
        }

        return config


    def save(self, folder):
        """
        Save the model to the specified folder.

        Args:
            folder (str): Path to the directory where the model should be saved.
        """
        model_dir = Path(folder) / "value_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights:bool=True):
        """
        Load a value model from a saved configuration.

        Args:
            config_path (str): Path to the configuration file.
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            ValueModel: Loaded value model instance.
        """
        model_dir = Path(config_path) / "value_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # Determine the wrapper type from the environment configuration
        # wrapper_dict = json.loads(config['env'])
        # wrapper_type = wrapper_dict.get("type")
        # if wrapper_type == 'GymnasiumWrapper':
        #     env = GymnasiumWrapper(wrapper_dict.get("env"))
        # else:
        #     raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

        env = EnvWrapper.from_json(config.get("env"))

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_layer_kernel = config.get("output_layer_kernel"),
                    optimizer_params = config.get("optimizer_params"),
                    scheduler_params = config.get("scheduler_params", None),
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model

class ActorModel(Model):
    
    def __init__(self,
                 env: EnvWrapper,
                 layer_config: List[Dict],
                 output_layer_kernel: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
                 optimizer_params: dict={'type':'Adam', 'params':{'lr':0.001}},
                 lr_scheduler: ScheduleWrapper=None,
                 device: str=None
                 ):
        super().__init__(env, layer_config, optimizer_params, lr_scheduler, device)
        self.output_config = output_layer_kernel

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'actor_mu': nn.LazyLinear(self.env.single_action_space.shape[-1]),
            'actor_pi': nn.Tanh()
        })

        # Move the model to the specified device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

    def forward(self, x, goal=None):
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)

        for layer in self.layers.values():
            x = layer(x)

        mu = self.output_layer["actor_mu"](x)
        pi = self.output_layer["actor_pi"](mu)
        pi = pi * T.tensor(self.env.single_action_space.high, dtype=T.float32, device=self.device)
        return mu, pi

    def get_config(self):
        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel':self.output_config,
            'optimizer_params': self.optimizer_params,
            'lr_scheduler': self.lr_scheduler.get_config(),
            'device': self.device.type,
        }

        return config


    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        if device:
            device = get_device(device)
        else:
            device = self.device

        #DEBUG
        print(f'lr_scheduler:{self.lr_scheduler}')
        print(f'lr_scheduler config:{self.lr_scheduler.get_config()}')

        env = GymnasiumWrapper(self.env.env_spec, self.env.wrappers)
        cloned_model = ActorModel(
            env=env,
            layer_config=self.layer_config.copy(),
            output_layer_kernel=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            device=device
        )
        
        if copy_weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())

        return cloned_model


    def save(self, folder):
        """
        Save the model to the specified folder.

        Args:
            folder (str): Path to the directory where the model should be saved.
        """
        model_dir = Path(folder) / "actor_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config, load_weights=True):
        """
        Load an actor model from a saved configuration.

        Args:
            config (dict): Configuration dictionary.
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            ActorModel: Loaded actor model instance.
        """
        # model_dir = Path(config_path) / "actor_model"
        # config_path = model_dir / "config.json"
        # model_path = model_dir / 'pytorch_model.onnx'

        # if config_path.is_file():
        #     with open(config_path, "r", encoding="utf-8") as f:
        #         config = json.load(f)
        # else:
        #     raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        env = EnvWrapper.from_json(config.get("env"))
        lr_scheduler = ScheduleWrapper(config.get("lr_scheduler", None))

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_layer_kernel = config.get("output_layer_kernel"),
                    # goal_shape = config.get("goal_shape", None)
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            try:
                model_path = Path(config.get("save_dir")) / "actor_model" / "pytorch_model.pt"
                model.load_state_dict(T.load(model_path, map_location=model.device))
            except Exception as e:
                print(f"Error loading model: {e}")

        return model


class CriticModel(Model):
    def __init__(self,
                 env: EnvWrapper,
                 state_layers: List[Dict],
                 merged_layers: List[Dict],
                 output_layer_kernel: [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
                #  goal_shape: tuple=None,
                 optimizer_params: dict={'type':'Adam', 'params':{'lr':0.001}},
                 lr_scheduler: ScheduleWrapper=None,
                 device: str=None
                 ):
        super().__init__(env, state_layers, optimizer_params, lr_scheduler, device)
        self.env = env
        # self.state_config = state_layers # Stored as layer config in parent
        self.merged_config = merged_layers
        self.output_config = output_layer_kernel
        # self.goal_shape = goal_shape

        # instantiate ModuleDicts for merged and Modules
        self.merged_layers = nn.ModuleDict()
        # self.output_layer = nn.ModuleDict()

        # set internal attributes
        for i, layer_info in enumerate(self.merged_config):
            layer_type = layer_info['type']
            layer_params = layer_info.get('params', {})
            self.merged_layers[f'{layer_type}_{i}'] = self._build_layer(layer_type, layer_params)
            
        # add module to model
        # self.add_module('merged_layers', self.merged_layers)

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'state_action_value': nn.LazyLinear(1)
        })
        # self.add_module('critic_output_layer', self.output_layer)

         # Move the model to the specified device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.merged_layers, self.merged_config)
        self._init_model(self.output_layer, self.output_config)

    def forward(self, state, action, goal=None):
        state = state.to(self.device)
        action = action.to(self.device)
        #DEBUG
        # print(f'critic state input shape:{state.size()}')
        # print(f'critic action input shape:{action.size()}')
        if goal is not None:
            goal = goal.to(self.device)
            state = T.cat([state, goal], dim=-1)

        # if self.goal_shape is not None:
            # state = T.cat([state, goal], dim=-1)

        for layer in self.layers.values():
            state = layer(state)
            #DEBUG
            # print(f'critic {layer} output shape:{state.size()}')

        merged = T.cat([state, action], dim=-1)
        #DEBUG
        # print(f'critic merged shape:{merged.size()}')
        for layer in self.merged_layers.values():
            merged = layer(merged)
            #DEBUG
            # print(f'critic {layer} output shape:{merged.size()}')

        for layer in self.output_layer.values():
            output = layer(merged)
            #DEBUG
            # print(f'critic {layer} output shape:{output.size()}')
        
        return output

    def get_config(self):
        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers) + len(self.merged_layers),
            'state_layers': self.layer_config,
            'merged_layers': self.merged_config,
            'output_layer_kernel': self.output_config,
            # 'goal_shape': self.goal_shape,
            'optimizer_params': self.optimizer_params,
            'lr_scheduler': self.lr_scheduler.get_config(),
            'device': self.device.type,
        }

        return config
    
    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        if device:
            device = get_device(device)
        else:
            device = self.device

        env = GymnasiumWrapper(self.env.env_spec)
        cloned_model = CriticModel(
            env=env,
            state_layers=self.layer_config.copy(),
            merged_layers=self.merged_config.copy(),
            output_layer_kernel=self.output_config.copy(),
            # goal_shape=self.goal_shape.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            device=device
        )
        
        if copy_weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())
            
            # # Optionally, clone the optimizer (requires more manual work, shown below)
            # cloned_optimizer = type(self.optimizer)(cloned_model.parameters(), **self.optimizer.defaults)
            # cloned_optimizer.load_state_dict(self.optimizer.state_dict())

        return cloned_model


    def save(self, folder):
        """
        Save the model to the specified folder.

        Args:
            folder (str): Path to the directory where the model should be saved.
        """
        model_dir = Path(folder) / "critic_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config, load_weights=True):
        """
        Load a critic model from a saved configuration.

        Args:
            config_path (str): Path to the configuration file.
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            CriticModel: Loaded critic model instance.
        """
        # model_dir = Path(config_path) / "critic_model"
        # config_path = model_dir / "config.json"
        # model_path = model_dir / 'pytorch_model.onnx'

        # if config_path.is_file():
        #     with open(config_path, "r", encoding="utf-8") as f:
        #         config = json.load(f)
        # else:
        #     raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        env = EnvWrapper.from_json(config.get("env"))
        lr_scheduler = ScheduleWrapper(config.get("lr_scheduler", None))
        
        model = cls(env = env,
                    state_layers = config.get("state_layers"),
                    merged_layers = config.get("merged_layers"),
                    output_layer_kernel = config.get("output_layer_kernel"),
                    # goal_shape = config.get("goal_shape", None)
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            try:
                model_path = Path(config.get("save_dir")) / "critic_model" / "pytorch_model.pt"
                model.load_state_dict(T.load(model_path, map_location=model.device))
            except Exception as e:
                print(f"Error loading model: {e}")

        return model


def build_layers(types: List[str], units_per_layer: List[int], initializers: List[str], kernel_params:List[dict]):
    """formats config into policy and value layers"""
    # get policy layers
    layers = []
    for type, units, kernel, k_param in zip(types, units_per_layer, initializers, kernel_params):
        layers.append({
            'type':type, 
            'params':{
                'units': units,
                'kernel': kernel,
                'kernel params': k_param
            }
        })
        
    return layers

def select_policy_model(env):
    """
    Select the appropriate policy model based on the environment's action space.

    Args:
        env (gym.Env): The environment object.

    Returns:
        Class: The class of the appropriate policy model.
    """
    #DEBUG
    # print(f'env action space type:{env.action_space}')
    # print(f'env observation space:{env.observation_space.shape}')
    # Check if the action space is discrete
    if isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete):
        model_class = StochasticDiscretePolicy
    # Check if the action space is continuous
    elif isinstance(env.action_space, gym.spaces.Box):
        model_class = StochasticContinuousPolicy
    else:
        raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")
    return model_class
