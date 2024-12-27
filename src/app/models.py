"""Holds Model classes used for Reinforcement learning."""

# imports
import json
import os
from typing import List, Tuple, Dict
from pathlib import Path
# import time

import torch as T
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical, Beta, Normal
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
import cnn_models
import torch_utils
from logging_config import logger
from env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper


class Model(nn.Module):
    """Base class for all RL models."""
    def __init__(self,
                 env:EnvWrapper,
                 layer_config,
                 optimizer_params: dict = {'type':'Adam', 'params':{'lr':0.001}},
                 scheduler_params:dict = None,
                 device=None):
        super(Model, self).__init__()
        self.env = env
        self.layer_config = layer_config
        self.layers = nn.ModuleDict()
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        if device is None:
            device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.device = device

        # Set initial input size based on environment observation space
        self.obs_space = self.env.observation_space
        if isinstance(self.env, GymnasiumWrapper):
            self.obs_space = self.env.single_observation_space
        # For CNNs, this will be 3D (channels, height, width)
        if len(self.obs_space.shape) == 3:  # For CNNs with images
            input_height, input_width, input_channels = self.obs_space.shape  # (channels, height, width)
            input_size = (input_channels, input_height, input_width)
        else:  # For dense layers, using flat observation vector
            input_size = np.prod(self.obs_space.shape[-1])

        # Build the layers dynamically based on config
        for i, layer_info in enumerate(self.layer_config):
            layer_type = layer_info['type']
            layer_params = layer_info.get('params', {})

            # Build the layer
            self.layers[f'{layer_type}_{i}'] = self._build_layer(layer_type, layer_params, input_size)

            # Update input size for the next layer based on the layer type
            if layer_type == 'cnn':
                kernel_size = layer_params.get('kernel_size', (3, 3))
                stride = layer_params.get('stride', 1)
                padding = layer_params.get('padding', 0)

                # Update height and width based on CNN formula
                input_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
                input_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1

                # Update the number of channels to out_channels
                input_channels = layer_params['out_channels']
                input_size = (input_channels, input_height, input_width)

            elif layer_type == 'pool':
                pool_size = layer_params.get('kernel_size', (2, 2))
                pool_stride = layer_params.get('stride', pool_size)

                # Update height and width after pooling
                input_height = (input_height - pool_size[0]) // pool_stride[0] + 1
                input_width = (input_width - pool_size[1]) // pool_stride[1] + 1
                input_size = (input_channels, input_height, input_width)

            elif layer_type == 'flatten':
                # Flatten the 3D output (channels, height, width) into a 1D vector
                input_size = input_channels * input_height * input_width

            elif layer_type == 'dense':
                input_size = layer_params['units']  # Units become the input size for the next layer
        # add module to model
        self.add_module('layers', self.layers)
        # Init optimizer and scheduler
        self.optimizer = self._init_optimizer()
        if self.scheduler_params:
            self.scheduler = self._init_scheduler(scheduler_params)
        else:
            self.scheduler = None

        # Move the model to the correct device
        self.to(self.device)

        return input_size

    def _build_layer(self, layer_type, params, input_size):
        if layer_type == 'dense':
            return nn.Linear(input_size, params['units'])
        elif layer_type == 'cnn':
            params['in_channels'] = input_size[0]  # Set input channels (e.g., 3 for RGB images)
            return nn.Conv2d(**params)
        elif layer_type == 'pool':
            return nn.MaxPool2d(**params)
        elif layer_type == 'dropout':
            return nn.Dropout(**params)
        elif layer_type == 'batchnorm':
            params['num_features'] = input_size[0]  # For CNN, num_features = in_channels
            return nn.BatchNorm2d(**params)
        elif layer_type == 'flatten':  # Adding flatten layer
            return nn.Flatten()
        elif layer_type == 'relu':
            return nn.ReLU()
        elif layer_type == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def _init_weights(self, layer_config, layers):
        # Loop through each layer config and corresponding layer
        #DEBUG
        # print(f'layer config:{layer_config}')
        # print(f'layers:{layers}')
        for config, (layer_name, layer) in zip(layer_config, layers.items()):
            # Check if the layer is one that supports weight initialization
            
            #DEBUG
            # print(config, type(config))
            # print(f"config type:{config['type']}")
            if config['type'] in ['dense', 'transformer']:
                init_config = config['params'].get('kernel', 'default')  # Get kernel init or 'default'

                # Apply the specified initialization scheme
                if init_config == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight, **config['params']['kernel params'])
                elif init_config == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight)
                elif init_config == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_config == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif init_config == 'truncated_normal':
                    nn.init.trunc_normal_(layer.weight, **config['params']['kernel params'])
                    nn.init.trunc_normal_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'uniform':
                    nn.init.uniform_(layer.weight, **config['params']['kernel params'])
                    nn.init.uniform_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'normal':
                    nn.init.normal_(layer.weight, **config['params']['kernel params'])
                    nn.init.normal_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'constant':
                    nn.init.constant_(layer.weight, **config['params']['kernel params'])
                    nn.init.constant_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'ones':
                    nn.init.ones_(layer.weight, **config['params']['kernel params'])
                    nn.init.ones_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'zeros':
                    nn.init.zeros_(layer.weight, **config['params']['kernel params'])
                    nn.init.zeros_(layer.bias, **config['params']['kernel params'])
                elif init_config == 'variance_scaling':
                    torch_utils.VarianceScaling_(layer.weight)
                elif init_config == 'default':
                    # Use PyTorch's default initialization (skip)
                    pass
                else:
                    raise ValueError(f"Unsupported initialization: {init_config}")

                # Optionally initialize the bias (if it exists)
                # if layer.bias is not None:
                #     nn.init.zeros_(layer.bias)

    def _init_optimizer(self):
        #DEBUG
        print(f'self._init_optimizer fired...')
        print(f'optimizer params: {self.optimizer_params}')
        print(f'Number of parameters: {len(list(self.parameters()))}')
        if self.optimizer_params['type'] == 'Adam':
            return optim.Adam(self.parameters(), **self.optimizer_params['params'])
        elif self.optimizer_params['type'] == 'SGD':
            return optim.SGD(self.parameters(), **self.optimizer_params['params'])
        elif self.optimizer_params['type'] == 'RMSprop':
            return optim.RMSprop(self.parameters(), **self.optimizer_params['params'])
        elif self.optimizer_params['type'] == 'Adagrad':
            return optim.Adagrad(self.parameters(), **self.optimizer_params['params'])
        else:
            raise NotImplementedError
    
    def _init_scheduler(self, scheduler):
        scheduler_type = scheduler.get('type', '').lower()
        scheduler_params = scheduler.get('params', {})
        
        if scheduler_type == 'cosineannealinglr':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        if scheduler_type == 'steplr':
            return optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        if scheduler_type == 'exponentiallr':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        


    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


    def get_config(self):
        pass


    def save(self, folder):
        pass


    @classmethod
    def load(cls, folder):
        pass


class StochasticDiscretePolicy(Model):
    """Policy model for predicting a probability distribution of a discrete action space.

    Attributes:
      env: OpenAI gym environment.
      layer_config: List of Dicts containing layer type, units, kernel initializers, and kernel params.
      output_layer_kernel: Dict of kernel initializer:params for the output layer.
      optimizer: Optimizer for training.
      optimizer_params: Dict of Parameter:value for the optimizer.
      learning_rate: Learning rate for the optimizer.
      distribution: Distribution returned by the policy ('categorical').
      device: Device to run the model on.

    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_layer_kernel: dict = {"default": {}},
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params:dict = None,
        distribution: str = 'categorical',
        device: str = 'cuda'
    ):
        input_size = super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel
        self.distribution = distribution

        # Get the action space of the environment
        self.act_space = self.env.action_space
        if isinstance(self.env, GymnasiumWrapper):
            self.act_space = self.env.single_action_space

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'policy_dense_output': nn.Linear(input_size, self.act_space.n)
        })

        # Initialize weights
        self._init_weights(self.layer_config, self.layers)
        self._init_weights(self.output_config, self.output_layer)

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        self.to(self.device)

    def forward(self, x):
        """Forward Propogation."""
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
        """Get model config."""

        config = {
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'distribution': self.distribution,
            'device': self.device,
        }

        return config

    def save(self, folder):
        # Ensure the model directory exists
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
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # Determine which wrapper to use
        # Convert json to dict first
        wrapper_dict = json.loads(config['env'])
        wrapper_type = wrapper_dict.get("type")
        if wrapper_type == 'GymnasiumWrapper':
            env = GymnasiumWrapper(wrapper_dict.get("env"))
        else:
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

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
            model.load_state_dict(T.load(model_path))

        return model

class StochasticContinuousPolicy(Model):
    """Policy model for predicting a probability distribution of a continuous action space.

    Attributes:
      env: OpenAI gym environment.
      layer_config: List of Dicts containing layer type, units, kernel initializers, and kernel params.
      output_layer_kernel: Dict of kernel initializer:params for the output layer.
      optimizer: Optimizer for training.
      optimizer_params: Dict of Parameter:value for the optimizer.
      learning_rate: Learning rate for the optimizer.
      distribution: Distribution returned by the policy ('beta' or 'normal').
      device: Device to run the model on.

    """

    def __init__(
        self,
        env:EnvWrapper,
        layer_config: List[Dict],
        output_layer_kernel: dict = {"default": {}},
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params:dict = None,
        distribution: str = 'beta',
        device: str = 'cuda'
    ):
        input_size = super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel
        self.distribution = distribution
        
        # Get the action space of the environment
        self.act_space = self.env.action_space
        if isinstance(self.env, GymnasiumWrapper):
            self.act_space = self.env.single_action_space
        # Create the output layer
        # input_size = np.prod(env.observation_space.shape[-1])
        self.output_layer = nn.ModuleDict({
            'policy_dense_output': nn.Linear(input_size, 2 * self.act_space.shape[-1])
        })

        # Initialize weights
        self._init_weights(self.layer_config, self.layers)
        self._init_weights(self.output_config, self.output_layer)

        # Initialize optimizer # Done in Parent
        # self.optimizer = self._init_optimizer()

        self.to(self.device)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        x = x.to(self.device)
        for layer in self.layers.values():
            x = layer(x)
            # print(f'x shape for layer {layer}: {x.shape}')
        x = self.output_layer['policy_dense_output'](x)

        # Split x into param1 and param2 for beta distribution
        param1, param2 = T.split(x, self.act_space.shape[-1], dim=-1)

        if self.distribution == 'beta':
            alpha = F.relu(param1) + 1.0
            beta = F.relu(param2) + 1.0
            dist = Beta(alpha, beta)
            return dist, alpha, beta
        elif self.distribution == 'normal':
            mu = param1
            sigma = F.softplus(param2)
            dist = Normal(mu, sigma)
            return dist, mu, sigma
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

    def get_config(self):
        """Get model config."""

        config = {
            # 'env': self.env.spec.id,
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'distribution': self.distribution,
            'device': self.device,
        }

        return config

    def save(self, folder):
        # Ensure the model directory exists
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
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # Determine which wrapper to use
        # Convert json to dict first
        wrapper_dict = json.loads(config['env'])
        wrapper_type = wrapper_dict.get("type")
        if wrapper_type == 'GymnasiumWrapper':
            env = GymnasiumWrapper(wrapper_dict.get("env"))
        else:
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

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
    """Value model for predicting state values.

    Attributes:
      env: OpenAI gym environment.
      layer_config: List of Tuples containing dense layer sizes, activations, and kernel initializers.
      output_layer_kernel: Dict of kernel initializer:params for the output layer.
      optimizer_params: Dict of type and parameters for the optimizer. See default value
      scheduler_params: Dict of type and params for torch learning rate schedulers
      device: Device to run the model on.

    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_layer_kernel: dict = {"default":{}},
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        scheduler_params = None,
        device = 'cuda'
    ):
        input_size = super().__init__(env, layer_config, optimizer_params, scheduler_params, device)
        self.output_config = output_layer_kernel

        # Create the output layer
        # input_size = np.prod(env.observation_space.shape[-1])
        self.output_layer = nn.ModuleDict({
            'value_dense_output': nn.Linear(input_size, 1)
        })

        # Initialize weights
        self._init_weights(self.layer_config, self.layers)
        self._init_weights(self.output_config, self.output_layer)

        # Initialize optimizer # Done in Parent
        # self.optimizer = self._init_optimizer()

        self.to(self.device)

    def forward(self, x):
        """Forward Propogation."""
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        x = x.to(self.device)
        for layer in self.layers.values():
            x = layer(x)

        x = self.output_layer['value_dense_output'](x)

        return x


    def get_config(self):
        """Get model config."""

        config = {
            # 'env': self.env.spec.id,
            "env": self.env.to_json(),
            'num_layers': len(self.layers),
            'layer_config': self.layer_config,
            'output_layer_kernel': self.output_config,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'device': self.device,
        }

        return config


    def save(self, folder):
        # Ensure the model directory exists
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
        model_dir = Path(config_path) / "value_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.onnx'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")
        
        # Determine which wrapper to use
        # Convert json to dict first
        wrapper_dict = json.loads(config['env'])
        wrapper_type = wrapper_dict.get("type")
        if wrapper_type == 'GymnasiumWrapper':
            env = GymnasiumWrapper(wrapper_dict.get("env"))
        else:
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_layer_kernel = config.get("output_layer_kernel"),
                    optimizer_params = config.get("optimizer_params"),
                    scheduler_params = config.get("scheduler_params", None),
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            model.load_state_dict(T.load(model_path))

        return model

class ActorModel(Model):
    
    def __init__(self,
                 env,
                 cnn_model = None,
                 dense_layers: List[Tuple[int, str, dict]] = [(400,"relu",{"default":{}}),(300,"relu",{"default":{}})],
                 output_layer_kernel: dict = {"default":{}},
                 goal_shape: tuple = None,
                 optimizer: str = 'Adam',
                 optimizer_params: dict = {},
                 learning_rate = 0.001,
                 normalize_layers: bool = False,
                 device = None):
        super().__init__()
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.env = env
        self.layer_config = dense_layers
        self.output_config = output_layer_kernel
        self.cnn_model = cnn_model
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

        self.dense_layers = nn.ModuleDict()
        self.output_layer = nn.ModuleDict()
        self.output_activation = nn.ModuleDict()
        
        # if self.cnn_model:
        #     obs_shape = env.observation_space.shape
        #     dummy_input = T.zeros(1, *obs_shape, device=self.device)
        #     dummy_input = dummy_input.permute(0, 3, 1, 2)
        #     cnn_output = self.cnn_model(dummy_input)
        #     input_size = cnn_output.view(cnn_output.size(0), -1).shape[1]
        # else:
        #     input_size = env.observation_space.shape[0]

        # Adding support for goal
        if self.cnn_model:
            # obs_shape = env.observation_space.shape
            dummy_input = T.zeros(1, *self._obs_space_shape, device=self.device)
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

        # add output layers to dicts
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
            x = T.cat([x, goal], dim=-1)

        for layer in self.dense_layers.values():
            x = layer(x)

        for layer in self.output_layer.values():
            mu = layer(x)
        
        for layer in self.output_activation.values():
            pi = layer(mu)
        
        # if self.clamp_output is not None:
        #     # print('clamp output fired...')
        #     pi = T.clamp(pi, -self.clamp_output, self.clamp_output) * T.tensor(self.env.action_space.high, dtype=T.float32, device=self.device)
        #     # print(f'pi: {pi}')
        #     return mu, pi
        
        # print('unclamped output fired')
        pi = pi * T.tensor(self.env.action_space.high, dtype=T.float32, device=self.device)
        # print(f'pi: {pi}')
        return mu, pi

    def get_config(self):
        config = {
            'env': self.env.spec.id,
            'cnn_model': self.cnn_model.get_config() if self.cnn_model is not None else None,
            'num_layers': len(self.dense_layers),
            'layer_config': self.layer_config,
            'output_layer_kernel':self.output_config,
            'goal_shape': self.goal_shape,
            'optimizer': self.optimizer.__class__.__name__,
            'optimizer_params': self.optimizer_params,
            'learning_rate': self.learning_rate,
            'normalize_layers': self.normalize_layers,
            'device': self.device,
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
            # clamp_output=self.clamp_output,
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
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights=True):
        model_dir = Path(config_path) / "policy_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.pt'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = gym.make(config.get("env"))
            cnn_model = config.get("cnn_model", None)
            if cnn_model:
                cnn_model = cnn_models.CNN(cnn_model['layers'], env)
            # dense_layers = config.get("dense_layers")
            # output_layer_kernel = config.get("output_layer_kernel")
            # goal_shape = config.get("goal_shape", None)
            # optimizer = config.get("optimizer")
            # optimizer_params = config.get("optimizer_params")
            # learning_rate = config.get("learning_rate")
            # normalize = config.get("normalize", False)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        actor_model = cls(
            env=env,
            cnn_model=cnn_model,
            dense_layers=config.get("dense_layers", [(400,"relu",{"default":{}}),(300,"relu",{"default":{}})]),
            output_layer_kernel=config.get("output_layer_kernel", {"default":{}}),
            goal_shape=config.get("goal_shape", None),
            optimizer=config.get("optimizer", "Adam"),
            optimizer_params=config.get("optimizer_params", {}),
            learning_rate=config.get("learning_rate", 0.001),
            normalize_layers=config.get("normalize_layers", False),
            device=config.get("device", None),
            )
        
        # Load weights if True
        if load_weights:
            actor_model.load_state_dict(T.load(model_path))

        return actor_model


class CriticModel(Model):
    def __init__(self,
                 env,
                 cnn_model = None,
                 state_layers: List[Tuple[int, str, dict]] = [(400,"relu",{"default":{}})],
                 merged_layers: list = [(300,"relu",{"default":{}})],
                 output_layer_kernel: dict = {"default":{}},
                 goal_shape: tuple = None,
                 optimizer: str = 'Adam',
                 optimizer_params: dict = {},
                 learning_rate = 0.001,
                 normalize_layers: bool = False,
                 device = None
                 ):
        super().__init__()
        self.device = device if device else T.device('cuda' if T.cuda.is_available() else 'cpu')
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
            dummy_input = T.zeros(1, *self._obs_space_shape, device=self.device)
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
            state = T.cat([state, goal], dim=-1)

        for layer in self.state_layers.values():
            state = layer(state)

        merged = T.cat([state, action], dim=-1)
        logger.debug(f"merged layer input: {merged}")
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
            'env': self.env.spec.id,
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
            'device': self.device,
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
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        config = self.get_config()

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


    @classmethod
    def load(cls, config_path, load_weights=True):
        model_dir = Path(config_path) / "value_model"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.pt'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            env = gym.make(config.get("env"))
            cnn_model = config.get("cnn_model", None)
            if cnn_model:
                cnn_model = cnn_models.CNN(cnn_model['layers'], env)
            # state_layers = config.get("state_layers")
            # merged_layers = config.get("merged_layers")
            # output_layer_kernel = config.get("output_layer_kernel")
            # goal_shape = config.get("goal_shape", None)
            # optimizer = config.get("optimizer")
            # learning_rate = config.get("learning_rate")
            # optimizer_params = config.get("optimizer_params")
            # normalize = config.get("normalize", False)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        model = cls(env=env,
                    cnn_model=cnn_model,
                    state_layers=config.get("state_layers", [(400,"relu",{"default":{}})]),
                    merged_layers=config.get("merged_layers", [(300,"relu",{"default":{}})]),
                    output_layer_kernel=config.get("output_layer_kernel", {"default":{}}),
                    goal_shape=config.get("goal_shape", None),
                    optimizer=config.get("optimizer", "Adam"),
                    optimizer_params=config.get("optimizer_params", {}),
                    learning_rate=config.get("learning_rate", 0.001),
                    normalize_layers=config.get("normalize", False),
                    device=config.get("device", None)
                    )
        
        # Load weights if True
        if load_weights:
            model.load_state_dict(T.load(model_path))

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
    # Check if the action space is discrete
    if isinstance(env.action_space, gym.spaces.Discrete):
        model_class = StochasticDiscretePolicy
    # Check if the action space is continuous
    elif isinstance(env.action_space, gym.spaces.Box):
        model_class = StochasticContinuousPolicy
    else:
        raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")
    return model_class
