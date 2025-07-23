from typing import Optional
import torch as T
from models import Model
import numpy as np
from env_wrapper import EnvWrapper
from schedulers import ScheduleWrapper
from logging_config import get_logger
import gymnasium as gym

class ICM(Model):
    """Intrinsic Curiousity Module."""
    def __init__(self, env:EnvWrapper, model_configs:dict, optimizer_params:dict, reward_weight:float=0.1,
                 reward_scheduler: Optional[ScheduleWrapper]=None, beta:float=0.2,
                 log_level: str = 'info', device:Optional[str | T.device]=None):
        try:
            super().__init__(env, [], optimizer_params, device=device)
            self.model_configs = model_configs
            self.reward_weight = reward_weight
            self.reward_scheduler = reward_scheduler
            self.beta = beta
            # Internal Attributes
            self._use_encoder = False

            # Instantiate model attributes
            self.encoder = None
            self.inverse_model = None
            self.forward_model = None

            # TODO: Move self.logger call to base model class
            self.logger = get_logger(__name__, log_level)

            # Determine action and observation space properties
            action_space = (self.env.single_action_space
                            if hasattr(self.env, "single_action_space")
                            else self.env.action_space)
            self._is_discrete = isinstance(action_space, gym.spaces.Discrete)
            self.action_dim = (int(action_space.n),) if self._is_discrete else action_space.shape

            obs_space = (self.env.single_observation_space
                        if hasattr(self.env, "single_observation_space")
                        else self.env.observation_space)
            if isinstance(obs_space, gym.spaces.Dict):
                self.obs_dim = obs_space['observation'].shape
            else:
                self.obs_dim = obs_space.shape

            # Remove the empty 'layers' ModuleDict inherited from the base class
            if hasattr(self, 'layers') and not any(self.layers.parameters()):
                del self.layers

            # Initialize models
            self._init_model()

            # Initialize model weights
            if self._use_encoder:
                self._init_weights(self.model_configs['encoder']['layer_config'], self.encoder)
                self._init_weights(self.model_configs['encoder']['output_layer'], self.encoder)
            self._init_weights(self.model_configs['inverse_model']['layer_config'], self.inverse_model)
            self._init_weights(self.model_configs['inverse_model']['output_layer'], self.inverse_model)
            self._init_weights(self.model_configs['forward_model']['layer_config'], self.forward_model)
            self._init_weights(self.model_configs['forward_model']['output_layer'], self.forward_model)

            # Initialize Optimizer
            self.optimizer = self._init_optimizer()

            # Move to device
            self.to(self.device)

        except Exception as e:
            self.logger.error(f"Error in ICM init: {e}", exc_info=True)

    def _init_model(self)->None:
        """
        Initializes models according to config
        """
        try:
            # Loop over model keys in config
            for model_name, model_config in self.model_configs.items():
                model = T.nn.ModuleDict()
                # Build hidden layers
                for i, layer_info in enumerate(model_config['layer_config']):
                    layer = self._build_layer(layer_info['type'], layer_info.get('params', {}).copy())
                    model[f"{model_name}_{layer_info['type']}_{i}"] = layer

                # Build output layer/s
                # for i, output_info in enumerate(model_config['output_layer']):
                output_info = model_config['output_layer'][0]
                if model_name == 'inverse_model':
                    output = T.nn.LazyLinear(output_info.get('params', {}).get('units', int(np.prod(self.action_dim))))
                else:
                    output = T.nn.LazyLinear(output_info.get('params', {}).get('units', int(np.prod(self.obs_dim))))
                model[f"{model_name}_{output_info['type']}_output"] = output

                if model_name == 'encoder':
                    self.encoder = model
                    self.encoder.to(self.device)
                    self._use_encoder = True
                    self.add_module('encoder', self.encoder)
                elif model_name == 'inverse_model':
                    self.inverse_model = model
                    self.inverse_model.to(self.device)
                    self.add_module('inverse_model', self.inverse_model)
                elif model_name == 'forward_model':
                    self.forward_model = model
                    self.forward_model.to(self.device)
                    self.add_module('forward_model', self.forward_model)

        except Exception as e:
            self.logger.error(f"Error in _build_submodel: {e}", exc_info=True)
        with T.no_grad():
            dummy_state = T.ones((32, *self.obs_dim), device=self.device, dtype=T.float)

            # Encoder
            if self._use_encoder:
                s = self._forward_submodel(dummy_state, self.encoder)
                next_s = self._forward_submodel(dummy_state, self.encoder)
            else:
                s = dummy_state
                next_s = dummy_state

            # Action for inverse and forward models
            if self._is_discrete:
                action = T.randint(0, self.action_dim[0], (32,), device=self.device)
                action_input = T.nn.functional.one_hot(action.long(), num_classes=int(np.prod(self.action_dim)))
            else:
                action_input = T.randn(32, *self.action_dim, device=self.device)

            # Inverse model: [state, next state]
            inverse_input = T.cat([s, next_s], dim=-1)
            self._forward_submodel(inverse_input, self.inverse_model)

            # Forward model: [s, action]
            forward_input = T.cat([s, action_input], dim=-1)
            self._forward_submodel(forward_input, self.forward_model)

    def _forward_submodel(self, x, submodel):
        """Helper to forward pass through a submodel."""
        for name, layer in submodel.items():
            x = layer(x)
        return x

    def encode(self, state):
        """Feature Extractor."""
        if self._use_encoder:
            return self._forward_submodel(state, self.encoder)
        return state

    def forward(self, states, next_states, actions):
        """Run inference on Inverse and Forward models"""
        encoded_states = self.encode(states)
        encoded_next_states = self.encode(next_states)
        inverse_input = T.cat([encoded_states, encoded_next_states], dim=-1)
        pred_actions = self._forward_submodel(inverse_input, self.inverse_model)
        if self._is_discrete:
            actions = T.nn.functional.one_hot(actions.long(), num_classes=int(np.prod(self.action_dim))).float()
        forward_input = T.cat([encoded_states, actions], dim=-1)
        pred_next_states = self._forward_submodel(forward_input, self.forward_model)

        return pred_actions, pred_next_states, encoded_next_states

    def compute_intrinsic_reward(self, states, next_states, actions):
        """Computes and returns the Intrinsic Rewards"""
        with T.no_grad():
            _, pred_next_states, next_states = self.forward(states, next_states, actions)
            error = (pred_next_states - next_states).pow(2).sum(dim=-1)  # Squared L2 norm
            intrinsic_rewards = (self.reward_weight / 2) * error

            return intrinsic_rewards

    def train(self, states, next_states, actions):
        self.optimizer.zero_grad()
        pred_actions, pred_next_states, encoded_next_states = self.forward(states, next_states, actions)

        if self._is_discrete:
            inverse_loss = T.nn.CrossEntropyLoss()(pred_actions, actions)
        else:
            inverse_loss = T.nn.MSELoss()(pred_actions, actions)

        forward_loss = 0.5 * T.nn.MSELoss()(pred_next_states, encoded_next_states)
        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        loss.backward()
        self.optimizer.step()

        # Update scheduler
        if self.reward_scheduler:
            self.reward_scheduler.step()

        return loss.item()