from typing import Optional
import torch as T
from models import Model
import numpy as np
from env_wrapper import EnvWrapper
from schedulers import ScheduleWrapper
from logging_config import get_logger
import gymnasium as gym
from pathlib import Path
import json
import logging

class ICM(Model):
    """Intrinsic Curiousity Module."""
    def __init__(self, env:EnvWrapper, model_configs:dict, optimizer_params:dict, reward_weight:float=0.1,
                 reward_scheduler: Optional[ScheduleWrapper]=None, beta:float=0.2,
                 extrinsic_threshold: int=0, warmup:int=0, log_level: str = 'info',
                 device:Optional[str | T.device]=None):
        try:
            super().__init__(env, [], optimizer_params, device=device)
            self.model_configs = model_configs
            self.reward_weight = reward_weight
            self.reward_scheduler = reward_scheduler
            self.beta = beta
            self.extrinsic_threshold = extrinsic_threshold
            self.warmup = warmup
            # Internal Attributes
            self._use_encoder = False
            self._use_extrinsic = False

            # Instantiate model attributes
            self.encoder = None
            self.inverse_model = None
            self.forward_model = None

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
                encoder_config = self.model_configs['encoder']['layer_config'] + self.model_configs['encoder']['output_layer']
                self._init_weights(encoder_config, self.encoder)
            inverse_config = self.model_configs['inverse_model']['layer_config'] + self.model_configs['inverse_model']['output_layer']
            self._init_weights(inverse_config, self.inverse_model)
            forward_config = self.model_configs['forward_model']['layer_config'] + self.model_configs['forward_model']['output_layer']
            self._init_weights(forward_config, self.forward_model)

            # Initialize Optimizer
            self.optimizer = self._init_optimizer()

            # Move to device
            self.to(self.device)

            # Warmup models by training on synthetic data
            if self.warmup > 0:
                self._warmup_models()

        except Exception as e:
            self.logger.error(f"Error in ICM init: {e}", exc_info=True)

    def _warmup_models(self)->None:
        """
        Warmup models by training on synthetic data
        """
        # Determine observation and action space properties
        obs_space = (self.env.single_observation_space
                    if hasattr(self.env, "single_observation_space")
                    else self.env.observation_space)
        if isinstance(obs_space, gym.spaces.Dict):
            obs_high = T.tensor(obs_space['observation'].high, device=self.device).float()
        else:
            obs_high = T.tensor(obs_space.high, device=self.device).float()
        if obs_high.isinf().any():
            self.logger.warning("Observation space is unbounded, using default value of 10.0")
            obs_high = T.ones(self.obs_dim, device=self.device).float() * 10.0
            print(f'obs_high: {obs_high}')
        action_space = (self.env.single_action_space
                        if hasattr(self.env, "single_action_space")
                        else self.env.action_space)
        
        for _ in range(self.warmup):
            states = T.randn((512, *self.obs_dim), device=self.device, dtype=T.float) * obs_high
            next_states = T.randn((512, *self.obs_dim), device=self.device, dtype=T.float) * obs_high
            if self._is_discrete:
                action = T.randint(0, self.action_dim[0], (512,), device=self.device)
                action_input = T.nn.functional.one_hot(action.long(), num_classes=int(np.prod(self.action_dim)))
            else:
                action_high = T.tensor(action_space.high, device=self.device).float()
                action_input = T.randn(512, *self.action_dim, device=self.device) * action_high
            loss = self.train(states, next_states, action_input)
            # print(f'Warmup loss: {loss}')

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
        # x = x.to(self.device)
        for name, layer in submodel.items():
            x = layer(x)
        return x

    def encode(self, state):
        """Feature Extractor."""
        state = state.to(self.device)
        if self._use_encoder:
            return self._forward_submodel(state, self.encoder)
        return state

    def forward(self, states, next_states, actions):
        """Run inference on Inverse and Forward models"""
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        encoded_states = self.encode(states)
        encoded_next_states = self.encode(next_states)
        inverse_input = T.cat([encoded_states, encoded_next_states], dim=-1)
        pred_actions = self._forward_submodel(inverse_input, self.inverse_model)
        if self._is_discrete:
            actions = T.nn.functional.one_hot(actions.long(), num_classes=int(np.prod(self.action_dim)), device=self.device).float()
        forward_input = T.cat([encoded_states, actions], dim=-1)
        pred_next_states = self._forward_submodel(forward_input, self.forward_model)

        return pred_actions, pred_next_states, encoded_next_states

    def compute_intrinsic_reward(self, states, next_states, actions):
        """Computes and returns the Intrinsic Rewards"""
        with T.no_grad():
            _, pred_next_states, encoded_next_states = self.forward(states, next_states, actions)
            error = (pred_next_states - encoded_next_states).pow(2).sum(dim=-1)
            reward_weight = self.reward_weight
            if self.reward_scheduler:
                reward_weight *= self.reward_scheduler.get_factor()
            intrinsic_rewards = (reward_weight / 2) * error

            return intrinsic_rewards

    def train(self, states, next_states, actions):
        # Set models to train mode
        if self._use_encoder:
            self.encoder.train()
        self.inverse_model.train()
        self.forward_model.train()

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)

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

        # Set models to eval mode
        if self._use_encoder:
            self.encoder.eval()
        self.inverse_model.eval()
        self.forward_model.eval()

        return loss.item()

    def get_config(self):
        """Returns the configuration of the ICM model."""
        return {
            "env": self.env.to_json(),
            "model_configs": self.model_configs,
            "optimizer_params": self.optimizer_params,
            "reward_weight": self.reward_weight,
            "reward_scheduler": self.reward_scheduler.get_config() if self.reward_scheduler else None,
            "beta": self.beta,
            "extrinsic_threshold": self.extrinsic_threshold,
            "warmup": self.warmup,
            "log_level": logging.getLevelName(self.logger.getEffectiveLevel()).lower(),
            "device": self.device.type
        }

    def save(self, folder):
        """Save the model and its configuration."""
        model_dir = Path(folder) / "curiosity"
        model_dir.mkdir(parents=True, exist_ok=True)
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')
        config = self.get_config()
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, folder):
        """Load a model from a saved configuration."""
        model_dir = Path(folder) / "curiosity"
        config_path = model_dir / "config.json"
        model_path = model_dir / 'pytorch_model.pt'

        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No configuration file found in {config_path}")

        # Load EnvWrapper
        env_wrapper = EnvWrapper.from_json(config["env"])
        if config['reward_scheduler'] is not None:
            scheduler = ScheduleWrapper(config['reward_scheduler'])
        else:
            scheduler = None

        model = cls(env=env_wrapper, model_configs=config['model_configs'], optimizer_params=config['optimizer_params'],
                    reward_weight=config['reward_weight'], reward_scheduler=scheduler,
                    beta=config['beta'], extrinsic_threshold=config['extrinsic_threshold'], warmup=config['warmup'],
                    device=config['device'], log_level=config['log_level'])
        model.load_state_dict(T.load(model_path))
        return model