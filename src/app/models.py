"""Holds Model classes used for Reinforcement learning."""

# imports
from abc import abstractmethod
import json
import os
from typing import Optional, List, Tuple, Dict, Iterator
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import torch as T
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import torch.nn.functional as F
from torch.distributions import Distribution, TransformedDistribution, Independent, Categorical, Beta, Normal, Kumaraswamy


from app.distributions import SquashedNormal, ScaledBeta, ScaledKumaraswamy
from app.torch_utils import get_device, VarianceScaling_
from app.logging_config import get_logger
from app.env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from app.utils import check_for_inf_or_NaN
from app.schedulers import ScheduleWrapper

class Model(nn.Module):
    """
    Base class for all reinforcement learning models.

    This class dynamically constructs a neural network based on the provided layer configuration
    and supports various optimizers and learning rate schedulers.

    Attributes:
        env (EnvWrapper): The environment wrapper for the model.
        layer_config (list): List of dictionaries specifying the layers and their parameters.
        output_config (dict): Configuration for output layer initialization.
        optimizer_params (dict): Dictionary specifying optimizer type and parameters.
        lr_scheduler (ScheduleWrapper|None): Learning rate scheduler.
        device (str|None): The device ('cpu' or 'cuda') to run the model on (default: None = Cuda if available else CPU).
        log_level (str): Log level (default: 'info').
    """
    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_config: dict,
        optimizer_params: dict|None = None,
        lr_scheduler: ScheduleWrapper|None = None,
        device: str|None = None,
        # log_level: str = 'info'
    ):
        """
        Sets up the module dictionary of layers (most of which
        will be lazy).

        Args:
            env (EnvWrapper): Environment wrapper.
            layer_config (list): List of dictionaries specifying the layers and params.
            output_config (dict): Configuration for output layer initialization.
            optimizer_params (dict): Optimizer configuration.
            lr_scheduler (ScheduleWrapper|None): LR scheduler configuration.
            device (str|None): Device to run on (default: None = Cuda if available else CPU).
            log_level (str): Log level (default: 'info').
        """
        super().__init__()
        self.env = env
        self.layer_config = layer_config
        self.output_config = output_config
        self.layers = nn.ModuleDict()
        self.optimizer_params = optimizer_params or {'type': 'Adam', 'params': {'lr': 0.001}}
        self.lr_scheduler = lr_scheduler
        self.device = get_device(device)
        self.logger = get_logger(self.__class__.__name__, level='INFO')

        # Set references to env action and observation spaces
        self.obs_space = (self.env.single_observation_space if hasattr(self.env, "single_observation_space") 
                        else self.env.observation_space)
        self.act_space = (self.env.single_action_space 
                          if hasattr(self.env, "single_action_space") 
                          else self.env.action_space)

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

        # Dry run forward pass to initialize lazy modules
        # Check if the observation space is a dictionary AND contains goal-conditioned keys
        is_goal_conditioned = (isinstance(self.obs_space, gym.spaces.Dict) and 
                              self.env.obs_key in self.obs_space.spaces and 
                              self.env.goal_key in self.obs_space.spaces)
        
        if is_goal_conditioned:
            obs_shape = self.obs_space[self.env.obs_key].shape
            goal_shape = self.obs_space[self.env.goal_key].shape
            state_input = T.ones((32, *obs_shape), device=self.device, dtype=T.float)
            goal_input = T.ones((32, *goal_shape), device=self.device, dtype=T.float)
            # Check if CriticModel instance to pass action dummy values
            if isinstance(self, ContinuousCritic):
                action_shape = self.env.single_action_space.shape
                action_input = T.ones((32, *action_shape), device=self.device, dtype=T.float)
                with T.no_grad():
                    _ = self.forward(state_input, action_input, goal_input)
            else:
                with T.no_grad():
                    _ = self.forward(state_input, goal_input)
        else:
            # Handle both regular Box spaces and non-goal-conditioned Dict spaces
            if isinstance(self.obs_space, gym.spaces.Dict):
                if self.env.obs_key in self.obs_space.spaces:
                    obs_shape = self.obs_space.spaces[self.env.obs_key].shape
            else:
                obs_shape = self.obs_space.shape
            state_input = T.ones((32, *obs_shape), device=self.device, dtype=T.float)
            if isinstance(self, ContinuousCritic):
                action_shape = self.env.single_action_space.shape
                action_input = T.ones((32, *action_shape), device=self.device, dtype=T.float)
                with T.no_grad():
                    _ = self.forward(state_input, action_input)
            else:
                with T.no_grad():
                    _ = self.forward(state_input)

        # Initialize weights after lazy modules are materialized
        self._init_weights(layer_config, module_dict)

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
            kernel_params = config.get('params', {}).get('kernel_params', {}) # Get kernel params or empty dict
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

    def _init_optimizer(self, parameters: Iterator[Parameter] | None = None):
        """
        Initialize the optimizer for the model.

        Args:
            parameters (Iterator[Parameter] | None): Iterator over the parameters to optimize. If None, uses all parameters.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if parameters is None:
            parameters = self.parameters()
        original_optimizer_type = self.optimizer_params['type']
        optimizer_type = str(original_optimizer_type).lower()
        optimizer_params = self.optimizer_params['params']
        if optimizer_type == 'adam':
            return optim.Adam(parameters, **optimizer_params)
        elif optimizer_type == 'sgd':
            return optim.SGD(parameters, **optimizer_params)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(parameters, **optimizer_params)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(parameters, **optimizer_params)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {original_optimizer_type}")
    
    def _preprocess_state(self, state):
        """
        Preprocess the state tensor to handle various shapes, including flat vectors and images.
        
        - Adds a feature dim to 1D (flat) states.
        - Adds a channel dim to 3D (grayscale image) states.
        - Permutes image states from Gymnasium envs if needed (HWC -> CHW).
        
        Returns:
            Tensor: Preprocessed state.
        """
        # Handle flat (1D) states by adding a feature dimension (e.g., for single-feature observations)
        if state.dim() == 1:
            state = state.unsqueeze(-1)  # Reshape to (batch_size, 1)

        # Handle grayscale image states without channel dim (e.g., (batch_size, height, width) -> (batch_size, 1, height, width))
        if state.dim() == 3:
            state = state.unsqueeze(1)

        # Handle image-like observations from Gymnasium envs
        if isinstance(self.env, GymnasiumWrapper):
            # Permute color images from (B, H, W, C) to (B, C, H, W) if channels are last
            if state.dim() == 4 and state.shape[-1] in [3, 4]:
                state = state.permute(0, 3, 1, 2)

        return state

    def _unwrap_distribution(self, dist: Distribution) -> Distribution:
        """
        Recursively unwrap a distribution to get the base distribution (Normal, Beta, etc.).

        Args:
            dist (Distribution): The distribution to unwrap.

        Returns:
            Distribution: The base distribution.
        """
        while True:
            if isinstance(dist, Independent):
                dist = dist.base_dist
            elif isinstance(dist, (SquashedNormal, ScaledBeta, ScaledKumaraswamy)):
                dist = dist.base_dist
            elif isinstance(dist, TransformedDistribution):
                dist = dist.base_dist
            else:
                break
        return dist

    def get_mean_actions(self, dist: Distribution)->T.Tensor:
        """
        Get the mean action of the Transformed distribution.

        Args:
            dist (Distribution): The Transformed distribution to get the mean of.

        Returns:
            Tensor: The mean action of the Transformed distribution.
        """
        base_dist = self._unwrap_distribution(dist)

        if isinstance(base_dist, (Normal, Beta, Kumaraswamy)):
            # Get the low and high bounds of the action space
            low = T.tensor(self.env.single_action_space.low, dtype=T.float32, device=self.device)
            high = T.tensor(self.env.single_action_space.high, dtype=T.float32, device=self.device)

            if isinstance(base_dist, (Beta, Kumaraswamy)):
                return low + (high - low) * base_dist.mean
            elif isinstance(base_dist, Normal):
                mu = base_dist.loc
                scale = (high - low) / 2.0
                loc = (high + low) / 2.0
                return loc + scale * T.tanh(mu)
        elif isinstance(base_dist, Categorical):
            return base_dist.mode
        else:
            raise ValueError(f"Unsupported distribution: {type(base_dist)}")
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        return {
            "env": self.env.to_json(),
            'layer_config': self.layer_config,
            'output_config': self.output_config,
            'optimizer_params': self.optimizer_params,
            'lr_scheduler': self.lr_scheduler.get_config() if self.lr_scheduler else None,
            'device': self.device.type,
        }

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        if device:
            device = get_device(device)
        else:
            device = self.device

        if isinstance(self.env, IsaacSimWrapper):
            env = self.env  # Reuse existing instance
        else:
            env = EnvWrapper.from_json(self.env.to_json())

        return device, env

    def save(self, config_dir: Path | str, model_name: str):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save.
        """
        # Ensure the model directory exists
        model_dir = Path(config_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model parameters
        T.save(self.state_dict(), model_dir / 'pytorch_model.onnx')
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')

        # Save the model configuration
        config = self.get_config()
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, config_dir: Path | str, model_name: str, load_weights: bool = True, env: EnvWrapper | None = None):
        model_dir = Path(config_dir) / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} not found")
        config = json.load(open(model_dir / 'config.json'))
        if env is None:
            env = EnvWrapper.from_json(config.get("env"))
        lr_scheduler_config = config.get("lr_scheduler", None)
        lr_scheduler = ScheduleWrapper(**lr_scheduler_config) if lr_scheduler_config else None

        return config, lr_scheduler, env


class StochasticDiscretePolicy(Model):
    """
    Policy model for predicting a probability distribution over a discrete action space.

    This class builds on the `Model` base class and adds functionality specific to
    policies with a discrete action space, such as using a Categorical distribution
    for action selection.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list[dict]): Configuration of hidden layers (default: [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]).
        output_config (list[dict]): Configuration of the output layer weights (default: {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}).
        optimizer_params (dict): Parameters for the optimizer (default: {'type': 'Adam', 'params': {'lr': 0.001}}).
        lr_scheduler (ScheduleWrapper): Parameters for the learning rate scheduler (default: None).
        distribution (str): Type of distribution for action selection ['categorical'] (default: 'categorical').
        temperature (float): Temperature for the relaxed categorical distribution (default: 1.0).
        temperature_schedule (ScheduleWrapper, optional): Temperature scheduler configuration. Default=None
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        log_level (str): logger level. Default=info.
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: list[dict],
        output_config: list[dict] = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: ScheduleWrapper|None = None,
        distribution: str = 'categorical',
        temperature: float = 1.0,
        temperature_schedule: ScheduleWrapper|None = None,
        device: str|T.device|None = None,
    ):
        """
        Initialize the policy model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list[dict]): List of dictionaries specifying hidden layer configurations.
            output_config (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            lr_scheduler (ScheduleWrapper, optional): LR scheduler configuration. Default=None
            distribution (str): Type of distribution for actions ['categorical'] (default: 'categorical').
            temperature (float): Temperature for the relaxed categorical distribution (default: 1.0).
            temperature_schedule (ScheduleWrapper, optional): Temperature scheduler configuration. Default=None
            device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        """
        
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)
        self.distribution = distribution
        self.temperature = temperature
        self.temperature_schedule = temperature_schedule

        # Set reference to the number of actions in the environment
        self.num_actions = self.act_space.n

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'policy_dense_output': nn.LazyLinear(self.num_actions)
        })

        # Move to device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)
        
        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, x, goal=None):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).

        Returns:
            Tuple[Categorical, Tensor]: Action distribution and logits for the action space.
        """
        # Preprocess state to ensure correct formatting
        x = self._preprocess_state(x)
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)
            
        for layer in self.layers.values():
            x = layer(x)
        x = self.output_layer['policy_dense_output'](x)
        
        if self.distribution == 'categorical':
            temperature = self.temperature
            if self.temperature_schedule is not None:
                temperature *= self.temperature_schedule.get_factor()
            dist = Categorical(logits=x / temperature)
            return dist
        else:
            raise ValueError(f'Distribution {self.distribution} not supported.')

    def get_config(self):
        config = super().get_config()
        config.update({
            'distribution': self.distribution,
            'temperature': self.temperature,
            "temperature_schedule": self.temperature_schedule.get_config() if self.temperature_schedule is not None else None,
        })
        return config

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        device, env = super().clone(copy_weights, device)
        cloned_model = StochasticDiscretePolicy(
            env=env,
            layer_config=self.layer_config.copy(),
            output_config=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            distribution=self.distribution,
            temperature=self.temperature,
            temperature_schedule=self.temperature_schedule.clone() if self.temperature_schedule else None,
            device=device
        )
        if copy_weights:
            cloned_model.load_state_dict(self.state_dict())
        return cloned_model

    def save(self, config_dir: Path | str, model_name: str = "policy"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "policy").
        """
        return super().save(config_dir, model_name)

    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "policy", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load a policy model from a saved configuration.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to load (default: "policy_model").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            StochasticDiscretePolicy: Loaded policy model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_config = config.get("output_config", {"default":{}}),
                    optimizer_params = config.get("optimizer_params", {}),
                    lr_scheduler = lr_scheduler,
                    distribution = config.get("distribution", "categorical"),
                    temperature = config.get("temperature", None),
                    temperature_schedule = config.get("temperature_schedule", None),
                    device = config.get("device", None)
                    )

        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
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
        output_config (dict): Configuration of the output layer weights.
        optimizer_params (dict): Parameters for the optimizer.
        lr_scheduler (ScheduleWrapper, optional): LR scheduler configuration. Default=None
        distribution (str): Type of distribution for actions (default: 'beta').
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        # log_level (str): logger level. Default=info.
    """

    def __init__(
        self,
        env:EnvWrapper,
        layer_config: List[Dict],
        output_config: list[dict] = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: Optional[ScheduleWrapper] = None,
        distribution: str = 'beta',
        device: str|T.device|None = None,
    ):
        """
        Initialize the policy model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list): List of dictionaries specifying hidden layer configurations.
            output_config (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            lr_scheduler (ScheduleWrapper, optional): LR scheduler configuration. Default=None
            distribution (str): Type of distribution for actions (normal or beta) (default: 'beta').
            # obs_key (str|None): Observation key (default: None).
            # goal_key (str | None): Goal key (default: None).
            device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
            # log_level (str): logger level. Default=info.
        """
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)
        self.distribution = distribution
        
        # Set lower/upper bounds of action space to Tensors
        # self.act_space_low = T.tensor(self.act_space.low, dtype=T.float32, device=self.device)
        # self.act_space_high = T.tensor(self.act_space.high, dtype=T.float32, device=self.device)
        # Set number of actions in the action space
        self.num_actions = self.act_space.shape[-1]
        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'policy_output_param_1': nn.LazyLinear(self.num_actions),
            'policy_output_param_2': nn.LazyLinear(self.num_actions),
        })

        # Move model to device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, x, goal=None):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).

        Returns:
            Distribution, Tensor, Tensor: Action distribution and its parameters.
        """
         # Preprocess state to ensure correct formatting
        x = self._preprocess_state(x)
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)

        #DEBUG
        # print(f'input x: {x}')

        for layer in self.layers.values():
            x = layer(x)
            #DEBUG
            # print(f'output x layer {layer}: {x}')

        param_1 = self.output_layer['policy_output_param_1'](x)
        param_2 = self.output_layer['policy_output_param_2'](x)

        # Check if parameters are finite
        if not T.isfinite(param_1).all() or not T.isfinite(param_2).all():
            # self.logger.warning(f'Non-finite parameters: {param_1}, {param_2}')
            param_1 = T.nan_to_num(param_1, nan=0.0, posinf=5.0, neginf=-5.0)
            param_2 = T.nan_to_num(param_2, nan=0.0, posinf=2.0, neginf=-10.0)

        if self.distribution in ['beta', 'kumaraswamy']:
            # Clamp params between -12 and 6 to allow max expressiveness within safe bounds of dist
            param_1 = T.clamp(param_1, min=-12, max=6)
            param_2 = T.clamp(param_2, min=-12, max=6)
            # softplus params to ensure >0 and add 1.0 for numerical stability
            alpha = F.softplus(param_1) + 1.0
            beta = F.softplus(param_2) + 1.0
            # Clamp alpha/beta to prevent exploding gradients
            alpha = T.clamp(alpha, min=1e-3, max=10.0)
            beta = T.clamp(beta, min=1e-3, max=10.0)

            if self.distribution == 'beta':
                dist = ScaledBeta(Beta(alpha, beta), low=self.env.single_action_space.low, high=self.env.single_action_space.high)
        
            elif self.distribution == 'kumaraswamy':
                dist = ScaledKumaraswamy(Kumaraswamy(alpha, beta), low=self.env.single_action_space.low, high=self.env.single_action_space.high)

        elif self.distribution == 'normal':
            mu = T.clamp(param_1, min=-10.0, max=10.0)
            log_std = T.clamp(param_2, min=-6, max=2)
            sigma = T.exp(log_std) + 1e-8

            # # If action space unbounded, return Torch Normal dist, else SquashedNormal
            low = T.tensor(self.act_space.low, device=self.device)
            high = T.tensor(self.act_space.high, device=self.device)

            # if T.isinf(high).any() or T.isinf(low).any():
            #     dist = Normal(mu, sigma)
            # else:
            dist = SquashedNormal(
                Normal(mu, sigma),
                low=low,
                high=high
            )
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

        return Independent(dist, reinterpreted_batch_ndims=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'distribution': self.distribution,
        })
        return config

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        device, env = super().clone(copy_weights, device)
        cloned_model = StochasticContinuousPolicy(
            env=env,
            layer_config=self.layer_config.copy(),
            output_config=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            distribution=self.distribution,
            device=device
        )
        if copy_weights:
            cloned_model.load_state_dict(self.state_dict())
        return cloned_model

    def save(self, config_dir: Path | str, model_name: str = "policy"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "policy").
        """
        return super().save(config_dir, model_name)

    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "policy", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load a policy model from a saved configuration.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to load (default: "policy").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            StochasticContinuousPolicy: Loaded policy model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_config = config.get("output_config", {"default":{}}),
                    optimizer_params = config.get("optimizer_params", {}),
                    lr_scheduler = lr_scheduler,
                    distribution = config.get("distribution", "beta"),
                    device = config.get("device", "cpu")
                    )

        # Load weights if True
        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model


class ValueModel(Model):
    """
    Value model for predicting state values.

    This class extends the `Model` base class to implement a neural network for value function approximation in reinforcement learning.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): Configuration of hidden layers (default: [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}]).
        output_config (dict): Configuration of the output layer weights (default: {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}).
        optimizer_params (dict): Parameters for the optimizer (default: {'type': 'Adam', 'params': {'lr': 0.001}}).
        lr_scheduler (ScheduleWrapper): Parameters for the learning rate scheduler (default: None).
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_config: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params:dict = {'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: Optional[ScheduleWrapper] = None,
        device: str|T.device|None = None,
        # log_level: str = 'info'
    ):
        """
        Initialize the value model.

        Args:
            env (EnvWrapper): The environment wrapper.
            layer_config (list): List of dictionaries specifying hidden layer configurations.
            output_config (dict): Configuration for output layer initialization (default: {}).
            optimizer_params (dict, optional): Optimizer parameters (default: Adam with lr=0.001).
            lr_scheduler (ScheduleWrapper, optional): learning rate Scheduler parameters (default: None).
            device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        """
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)

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

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, x, goal=None):
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor (e.g., observation from the environment).
            goal (Tensor, optional): Goal tensor (default: None).

        Returns:
            Tensor: Predicted state value.
        """

        # Preprocess state to ensure correct formatting
        x = self._preprocess_state(x)
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)

        for layer in self.layers.values():
            x = layer(x)

        x = self.output_layer['value_dense_output'](x)

        return x

    def get_config(self):
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        device, env = super().clone(copy_weights, device)

        cloned_model = ValueModel(
            env=env,
            layer_config=self.layer_config.copy(),
            output_config=self.output_config.copy(),
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


    def save(self, config_dir: Path | str, model_name: str = "value"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "value").
        """
        return super().save(config_dir, model_name)

    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "value", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load a value model from a saved configuration.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to load (default: "value").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            ValueModel: Loaded value model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_config = config.get("output_config"),
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )
        # Load weights if True
        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model

class ActorModel(Model):
    """
    Actor model for continuous action spaces.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): List of dictionaries specifying hidden layer configurations.
        output_config (dict): Configuration for output layer initialization (default: {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}).
        optimizer_params (dict): Parameters for the optimizer (default: {'type': 'Adam', 'params': {'lr': 0.001}}).
        lr_scheduler (ScheduleWrapper): Parameters for the learning rate scheduler (default: None).
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        # log_level (str): Log level (default: 'info').
    """
    
    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_config: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params: dict={'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: ScheduleWrapper|None = None,
        device: str|T.device|None = None,
        # log_level: str='info'
    ):
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)

        # Set lower/upper bounds of action space to Tensors
        self.act_space_low = T.tensor(self.act_space.low, dtype=T.float32, device=self.device)
        self.act_space_high = T.tensor(self.act_space.high, dtype=T.float32, device=self.device)
        self.num_actions = self.act_space.shape[-1]

        # Create the output layer
        self.output_layer = nn.ModuleDict({
            'actor_mu': nn.LazyLinear(self.num_actions),
            'actor_pi': nn.Tanh()
        })

        # Move the model to the specified device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, x, goal=None):
        x = self._preprocess_state(x)
        x = x.to(self.device)
        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)

        for layer in self.layers.values():
            x = layer(x)

        mu = self.output_layer["actor_mu"](x)
        pi = self.output_layer["actor_pi"](mu)
        if not T.isinf(self.act_space_high).any() and not T.isinf(self.act_space_low).any():
            # Map to actual [low,high] bounds of env
            pi = self.act_space_low + (pi + 1.0) * 0.5 * (self.act_space_high - self.act_space_low)
           
        return mu, pi

    def get_config(self):
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        device, env = super().clone(copy_weights, device)

        cloned_model = ActorModel(
            env=env,
            layer_config=self.layer_config.copy(),
            output_config=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            device=device
        )
        
        if copy_weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())

        return cloned_model


    def save(self, config_dir: Path | str, model_name: str = "policy"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "policy").
        """
        return super().save(config_dir, model_name)


    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "policy", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load an actor model from a saved configuration.

        Args:
            config_dir (Path | str): Path to the configuration directory.
            model_name (str): Name of the model to load (default: "policy").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            ActorModel: Loaded actor model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_config = config.get("output_config"),
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )

        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model

class BaseCritic(Model):
    """
    Base class for critic models.
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_config: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params: dict = {'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: ScheduleWrapper|None = None,
        device: str|T.device|None = None,
        # log_level: str='info'
        ):
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        return super().clone(copy_weights, device)

    def save(self, config_dir: Path | str, model_name: str = "critic"):
        return super().save(config_dir, model_name)

    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "critic", load_weights: bool = True, env: EnvWrapper | None = None):
        return super().load(config_dir, model_name, load_weights, env)

class ContinuousCritic(BaseCritic):
    """
    Critic model for continuous action spaces.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): List of dictionaries specifying hidden layer configurations for the state.
        merged_config (list): List of dictionaries specifying hidden layer configurations for the merged state and action.
        output_config (dict): Configuration for output layer initialization (default: {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}).
        optimizer_params (dict): Parameters for the optimizer (default: {'type': 'Adam', 'params': {'lr': 0.001}}).
        lr_scheduler (ScheduleWrapper): Parameters for the learning rate scheduler (default: None).
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        # log_level (str): Log level (default: 'info').
    """
    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        merged_config: List[Dict],
        output_config: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params: dict={'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: ScheduleWrapper|None = None,
        device: str|T.device|None = None,
        # log_level: str='info'  
    ):
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)
        self.merged_config = merged_config
        # self.output_config = output_layer_kernel

        # instantiate ModuleDicts for merged and Modules
        self.merged_layers = nn.ModuleDict()

        # set internal attributes
        for i, layer_info in enumerate(self.merged_config):
            layer_type = layer_info['type']
            layer_params = layer_info.get('params', {})
            self.merged_layers[f'{layer_type}_{i}'] = self._build_layer(layer_type, layer_params)

        # Create the output layer
        self.output_layer = nn.ModuleDict({'State_Action_value': nn.LazyLinear(1)})
        # self.add_module('critic_output_layer', self.output_layer)

         # Move the model to the specified device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.merged_layers, self.merged_config)
        self._init_model(self.output_layer, self.output_config)

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, state, action, goal=None):
         # Preprocess state to ensure correct formatting
        state = self._preprocess_state(state)
        state = state.to(self.device)
        action = action.to(self.device)
        
        if goal is not None:
            goal = goal.to(self.device)
            state = T.cat([state, goal], dim=-1)

        for layer in self.layers.values():
            state = layer(state)

        merged = T.cat([state, action], dim=-1)
        for layer in self.merged_layers.values():
            merged = layer(merged)

        for layer in self.output_layer.values():
            output = layer(merged)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'merged_config': self.merged_config,
        })

        return config
    
    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        device, env = super().clone(copy_weights, device)
            
        cloned_model = ContinuousCritic(
            env=env,
            layer_config=self.layer_config.copy(),
            merged_config=self.merged_config.copy(),
            output_config=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            device=device
        )
        
        if copy_weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())

        return cloned_model


    def save(self, config_dir: Path | str, model_name: str = "critic"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "critic").
        """
        return super().save(config_dir, model_name)


    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "critic", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load a continuous critic model from a saved configuration.

        Args:
            config_dir (Path | str): Path to the configuration directory.
            model_name (str): Name of the model to load (default: "critic").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            ContinuousCritic: Loaded continuous critic model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    merged_config = config.get("merged_config"),
                    output_config = config.get("output_config"),
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
            model.load_state_dict(T.load(model_path, map_location=model.device))

        return model

class DiscreteCritic(BaseCritic):
    """
    Critic model for discrete action spaces.

    Attributes:
        env (EnvWrapper): The environment wrapper.
        layer_config (list): List of dictionaries specifying hidden layer configurations for the state.
        output_config (dict): Configuration for output layer initialization (default: {'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}).
        optimizer_params (dict): Parameters for the optimizer (default: {'type': 'Adam', 'params': {'lr': 0.001}}).
        lr_scheduler (ScheduleWrapper): Parameters for the learning rate scheduler (default: None).
        device (str|T.device|None): Device to run the model on (default: None = Cuda if available else CPU).
        log_level (str): Log level (default: 'info').
    """
    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict],
        output_config: dict = [{'type': 'dense', 'params': {'kernel': 'default', 'kernel params':{}}}],
        optimizer_params: dict={'type':'Adam', 'params':{'lr':0.001}},
        lr_scheduler: ScheduleWrapper|None = None,
        device: str|T.device|None = None,
        # log_level: str='info'
    ):
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)
        # self.output_config = output_layer_kernel

        # Create the output layer
        self.output_layer = nn.ModuleDict({'Q_values': nn.LazyLinear(self.env.single_action_space.n)})
        # self.add_module('critic_output_layer', self.output_layer)

         # Move the model to the specified device
        self.to(self.device)

        # initialize params
        self._init_model(self.layers, self.layer_config)
        self._init_model(self.output_layer, self.output_config)

        # Now that parameters exist, create the optimizer
        self.optimizer = self._init_optimizer(self.parameters())
        
        # If lr scheduler, bind the optimizer to it
        if self.lr_scheduler is not None:
            self.lr_scheduler.attach_optimizer(self.optimizer)

    def forward(self, state, goal=None):
         # Preprocess state to ensure correct formatting
        state = self._preprocess_state(state)
        state = state.to(self.device)
        
        if goal is not None:
            goal = goal.to(self.device)
            state = T.cat([state, goal], dim=-1)

        for layer in self.layers.values():
            state = layer(state)

        for layer in self.output_layer.values():
            output = layer(state)
        
        return output

    def get_config(self):
        config = super().get_config()
        return config
    
    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        # Reconstruct the model from its configuration
        device, env = super().clone(copy_weights, device)
            
        cloned_model = DiscreteCritic(
            env=env,
            layer_config=self.layer_config.copy(),
            output_config=self.output_config.copy(),
            optimizer_params=self.optimizer_params.copy(),
            lr_scheduler=self.lr_scheduler.clone() if self.lr_scheduler else None,
            device=device
        )
        
        if copy_weights:
            # Copy the model weights
            cloned_model.load_state_dict(self.state_dict())

        return cloned_model


    def save(self, config_dir: Path | str, model_name: str = "critic"):
        """
        Save the model to the specified configuration directory.

        Args:
            config_dir (Path | str): Configuration directory.
            model_name (str): Name of the model to save (default: "critic").
        """
        return super().save(config_dir, model_name)


    @classmethod
    def load(cls, config_dir: Path | str, model_name: str = "critic", load_weights: bool = True, env: EnvWrapper | None = None):
        """
        Load a discrete critic model from a saved configuration.

        Args:
            config_dir (Path | str): Path to the configuration directory.
            model_name (str): Name of the model to load (default: "critic").
            load_weights (bool): Whether to load the model weights (default: True).

        Returns:
            DiscreteCritic: Loaded discrete critic model instance.
        """
        config, lr_scheduler, env = super().load(config_dir, model_name, load_weights, env)

        model = cls(env = env,
                    layer_config = config.get("layer_config"),
                    output_config = config.get("output_config"),
                    optimizer_params = config.get("optimizer_params"),
                    lr_scheduler = lr_scheduler,
                    device = config.get("device")
                    )

        # Load weights if True
        if load_weights:
            model_path = Path(config_dir) / model_name / "pytorch_model.pt"
            model.load_state_dict(T.load(model_path, map_location=model.device))

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

def select_policy_model(env: EnvWrapper):
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

def select_critic_model(env: EnvWrapper):
    """
    Select the appropriate critic model based on the environment's action space.
    """
    if isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete):
        model_class = DiscreteCritic
    elif isinstance(env.action_space, gym.spaces.Box):
        model_class = ContinuousCritic
    else:
        raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")
    return model_class
