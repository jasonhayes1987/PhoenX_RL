"""
intrinsic_motivation.py

Pluggable intrinsic motivation framework for PhoenX RL.

Class hierarchy:
    IntrinsicMotivation (ABC, inherits from Model)
        ├── ICM                      (Pathak et al. 2017)
        ├── RND                      (Burda et al. 2018)
        ├── EpisodicNovelty          (k-NN component of NGU; Badia et al. 2020)
        └── CompositeIntrinsicMotivation

Designed to drop into the existing PhoenX agent / trainer surface:
    - .compute_intrinsic_reward(states, next_states, actions) -> Tensor
    - .train(states, next_states, actions) -> loss Tensor
    - .reward_weight / .reward_scheduler / .extrinsic_threshold / ._use_extrinsic
    - .save(folder) / .load(folder, env=...)

New hooks (additive — wired through Trainer):
    - .on_episode_end(env_indices)        # stateful subclasses (EpisodicNovelty)
    - .add_to_normalizers(...)            # internal normalizer feeding
    - .update_normalizers()               # internal normalizer flush-to-running-stats
    - .set_normalizers_mode(context)      # train/eval mode propagation

Combination rules for CompositeIntrinsicMotivation:
    additive_combination, multiplicative_combination,
    ngu_combination, max_combination, weighted_sum_combination
"""

from abc import abstractmethod
from typing import Callable, List, Literal, Sequence, Any
from pathlib import Path
import json
import logging

import torch as T
import numpy as np
import gymnasium as gym

from .models import Model
from .env_wrapper import EnvWrapper, Observation
from .schedulers import ScheduleWrapper
from .normalizer import BaseNormalizer, RunningNorm, RewardNorm
from .logging_config import get_logger


# =============================================================================
# Type registry — used by IntrinsicMotivation.load() to dispatch to subclasses
# =============================================================================
_REGISTRY: dict[str, type] = {}


def register_intrinsic_motivation(cls):
    """Decorator: register an IntrinsicMotivation subclass for save/load."""
    _REGISTRY[cls.__name__] = cls
    return cls

class IntrinsicMotivation(Model):
    """
    Abstract base for any module that produces intrinsic rewards.

    Inherits from Model so subclasses get _build_layer / _init_weights /
    _init_optimizer / device handling for free. The base itself has no
    networks — it only deletes the empty `self.layers` ModuleDict that
    Model creates and provides shared bookkeeping.

    Common attributes (every subclass has these — the agent code reads them):
        reward_weight        : float scaling on the intrinsic reward
        reward_scheduler     : optional decay schedule for reward_weight
        extrinsic_threshold  : agent step below which extrinsic reward is
                               suppressed (intrinsic-only warmup)
        _use_extrinsic       : flag the agent flips based on the threshold

    Optional self-managed normalizers (RND uses both; ICM typically uses neither):
        intrinsic_reward_normalizer    : RewardNorm over intrinsic rewards
    """

    def __init__(
        self,
        env: EnvWrapper,
        optimizer_params: dict | None = None,
        reward_weight: float = 1.0,
        reward_scheduler: ScheduleWrapper | None = None,
        extrinsic_threshold: int = 0,
        reward_normalizer: RewardNorm | None = None,
        log_level: str = 'info',
        device: str | T.device | None = None,
    ):
        self.logger = get_logger(self.__class__.__name__, level=log_level.upper())
        # Model.__init__ wants (env, layers, optimizer_params, device)
        super().__init__(env, [], optimizer_params or {}, device=device)
        # Strip the empty ModuleDict Model created — subclasses build their own.
        if hasattr(self, 'layers') and not any(self.layers.parameters()):
            del self.layers

        # Shared attributes
        self.optimizer_params = optimizer_params
        self.reward_weight = reward_weight
        self.reward_scheduler = reward_scheduler
        self.extrinsic_threshold = extrinsic_threshold
        self.reward_normalizer = reward_normalizer
        self.log_level = log_level
        # Flag to indicate if the intrinsic motivation is online (i.e. needs to be updated on each step)
        self.is_online = False

        # Discover action / observation shapes (vec env aware)
        action_space = (self.env.single_action_space
                        if hasattr(self.env, 'single_action_space')
                        else self.env.action_space)
        self._is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = ((int(action_space.n),)
                           if self._is_discrete else action_space.shape)

        obs_space = (self.env.single_observation_space
                     if hasattr(self.env, 'single_observation_space')
                     else self.env.observation_space)
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_dim = obs_space[env.obs_key].shape
        else:
            self.obs_dim = obs_space.shape

    @abstractmethod
    def train(self, states, next_states, actions=None) -> T.Tensor:
        """One training step. Return scalar loss tensor."""
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """Serializable config (excluding tensor state)."""
        ...
        
    def use_extrinsic_reward(self, step: int) -> bool:
        """
        Return True if extrinsic reward should be used at the given step, False otherwise.
        """
        return step >= self.extrinsic_threshold

    def compute_learn_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        **kwargs: Any,
    ) -> T.Tensor:
        """
        Return learn reward of shape (batch,) — defaults to zero reward. Subclasses can override.
        
        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).

        Returns:
            Tensor of shape (batch,) containing the learn reward for each state.
        """
        return T.zeros(states.shape[0], device=self.device, dtype=T.float32)

    def compute_rollout_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        env_indices: T.Tensor | None = None,
    ) -> T.Tensor:
        """
        Return rollout reward of shape (batch,) — defaults to zero reward. Subclasses can override.
        
        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).
            env_indices: Tensor of shape (batch,) containing the environment indices.

        Returns:
            Tensor of shape (batch,) containing the rollout reward for each state.
        """
        return T.zeros(states.shape[0], device=self.device, dtype=T.float32)

    def on_episode_end(self, env_indices: T.Tensor) -> None:
        """
        Called by the Trainer for each parallel env that just terminated/truncated.

        Args:
            env_indices: 1-D Long tensor of env indices that finished this step.

        Returns:
            None
        """
        pass

    # def add_to_normalizers(self, obs: Observation) -> None:
    #     """
    #     Add relavent data to the normalizers.

    #     Args:
    #         obs: Observation to feed.
    #     """
    #     if self.obs_normalizer is not None:
    #         self.obs_normalizer.add(obs.states.to(device=self.obs_normalizer.device))
    #     if self.intrinsic_reward_normalizer is not None:
    #         dones = T.zeros_like(obs.intrinsic_rewards, dtype=T.bool)
    #         self.intrinsic_reward_normalizer.add(obs.intrinsic_rewards, dones)

    # def update_normalizers(self) -> None:
    #     if self.obs_normalizer is not None:
    #         self.obs_normalizer.update()
    #     if self.intrinsic_reward_normalizer is not None:
    #         self.intrinsic_reward_normalizer.update()

    def set_normalizers_mode(self, context: Literal['train', 'eval']) -> None:
        for n in [self.reward_normalizer]:
            if n is None:
                continue
            n.train() if context == 'train' else n.eval()

    def _feed_reward_normalizer(self, reward: T.Tensor) -> None:
        """
        Feed reward to reward normalizer and update if present.

        Args:
            reward: Tensor to feed.
        """
        if self.reward_normalizer is not None:
            dones = T.zeros(reward.shape, dtype=T.bool, device=self.device)
            self.reward_normalizer.add(reward, dones)
            self.reward_normalizer.update()

    def _normalize_reward(self, intrinsic_reward: T.Tensor) -> T.Tensor:
        """
        Normalize intrinsic reward through the reward_normalizer if present, else identity.

        Args:
            intrinsic_reward: Tensor to normalize.

        Returns:
            Normalized tensor or original tensor if no normalizer is present.
        """
        if self.reward_normalizer is not None:
            return self.reward_normalizer.normalize(intrinsic_reward)
        return intrinsic_reward

    def _scaled_reward_weight(self) -> float:
        w = self.reward_weight
        if self.reward_scheduler is not None:
            w *= self.reward_scheduler.get_factor()
        return w

    def _forward_submodel(self, x: T.Tensor, submodel: T.nn.ModuleDict) -> T.Tensor:
        for _, layer in submodel.items():
            x = layer(x)
        return x

    def save(self, folder) -> None:
        model_dir = Path(folder) / 'intrinsic_motivation'
        model_dir.mkdir(parents=True, exist_ok=True)
        T.save(self.state_dict(), model_dir / 'pytorch_model.pt')
        config = self.get_config()
        # Tag the type so load() can dispatch
        config['type'] = self.__class__.__name__
        with open(model_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f)
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(
                str(model_dir / 'reward_normalizer_state.pt'))

    @classmethod
    def create_instance(cls, im_type: str, **kwargs) -> 'IntrinsicMotivation':
        if im_type == 'ICM':
            return ICM(**kwargs)
        elif im_type == 'RND':
            return RND(**kwargs)
        elif im_type == 'EpisodicNovelty':
            return EpisodicNovelty(**kwargs)
        else:
            raise ValueError(f"Unknown intrinsic motivation type: {im_type}")

    @classmethod
    def load(cls, folder: str | Path, env: EnvWrapper | None = None) -> 'IntrinsicMotivation':
        """Dispatches to the correct subclass based on the saved 'type' field."""
        model_dir = Path(folder) / 'intrinsic_motivation'
        config_path = model_dir / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError(f"No config at {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        type_name = config.get('type')
        if type_name not in _REGISTRY:
            raise ValueError(f"Unknown intrinsic motivation type: {type_name}")
        return _REGISTRY[type_name]._load_impl(folder, config, env)

    @classmethod
    def _load_impl(cls, folder, config, env) -> 'IntrinsicMotivation':
        """Each subclass implements its own load. Default raises."""
        raise NotImplementedError(f"{cls.__name__} must implement _load_impl")

@register_intrinsic_motivation
class ICM(IntrinsicMotivation):
    """
    Intrinsic Curiosity Module (Pathak et al. 2017).

    Three sub-networks:
        encoder  φ : s -> features        (optional; if absent, raw obs are used)
        inverse  g : (φ(s), φ(s')) -> â   (predicts action; supplies gradient
                                           that shapes φ to encode controllable
                                           features only)
        forward  f : (φ(s), a) -> φ̂(s')   (predicts next features; its squared
                                           error in feature space *is* the
                                           intrinsic reward)

    Loss:
        L = (1 - β) * L_inv  +  β * L_fwd
    Intrinsic reward:
        r_i = (η/2) * || f(φ(s), a) - φ(s') ||²
    """

    def __init__(
        self,
        env: EnvWrapper,
        model_configs: dict,
        optimizer_params: dict,
        reward_weight: float = 0.1,
        reward_scheduler: ScheduleWrapper | None = None,
        beta: float = 0.2,
        extrinsic_threshold: int = 0,
        reward_normalizer: RewardNorm | None = None,
        log_level: str = 'info',
        device: str | T.device | None = None,
    ):
        try:
            super().__init__(
                env=env,
                optimizer_params=optimizer_params,
                reward_weight=reward_weight,
                reward_scheduler=reward_scheduler,
                extrinsic_threshold=extrinsic_threshold,
                reward_normalizer=reward_normalizer,
                log_level=log_level,
                device=device,
            )
            self.model_configs = model_configs
            self.beta = beta

            # Internal flags
            self._use_encoder = False

            # Network attributes
            self.encoder: T.nn.ModuleDict | None = None
            self.inverse_model: T.nn.ModuleDict | None = None
            self.forward_model: T.nn.ModuleDict | None = None

            # Build networks and run dummy forward to materialize LazyLinear shapes
            self._init_model()

            # Init weights per config
            if self._use_encoder:
                cfg = (self.model_configs['encoder']['layer_config']
                       + self.model_configs['encoder']['output_layer'])
                self._init_weights(cfg, self.encoder)
            inv_cfg = (self.model_configs['inverse_model']['layer_config']
                       + self.model_configs['inverse_model']['output_layer'])
            self._init_weights(inv_cfg, self.inverse_model)
            fwd_cfg = (self.model_configs['forward_model']['layer_config']
                       + self.model_configs['forward_model']['output_layer'])
            self._init_weights(fwd_cfg, self.forward_model)

            self.optimizer = self._init_optimizer()
            self.to(self.device)

        except Exception as e:
            self.logger.error(f"Error in ICM init: {e}", exc_info=True)
            raise

    def _init_model(self) -> None:
        for name, cfg in self.model_configs.items():
            module = T.nn.ModuleDict()
            for i, layer_info in enumerate(cfg['layer_config']):
                layer = self._build_layer(layer_info['type'],
                                          layer_info.get('params', {}).copy())
                module[f"{name}_{layer_info['type']}_{i}"] = layer
            output_info = cfg['output_layer'][0]
            if name == 'inverse_model':
                out_units = output_info.get('params', {}).get(
                    'units', int(np.prod(self.action_dim)))
            else:
                out_units = output_info.get('params', {}).get(
                    'units',
                    self.encoder['encoder_dense_output'].out_features
                    if self._use_encoder
                    else int(np.prod(self.obs_dim))
                )
            module[f"{name}_{output_info['type']}_output"] = T.nn.LazyLinear(out_units)

            module.to(self.device)
            if name == 'encoder':
                self.encoder = module
                self._use_encoder = True
                self.add_module('encoder', module)
            elif name == 'inverse_model':
                self.inverse_model = module
                self.add_module('inverse_model', module)
            elif name == 'forward_model':
                self.forward_model = module
                self.add_module('forward_model', module)

        # Dummy forward to materialize LazyLinear shapes
        with T.no_grad():
            dummy = T.ones((32, *self.obs_dim), device=self.device, dtype=T.float)
            s = self._forward_submodel(dummy, self.encoder) if self._use_encoder else dummy
            ns = self._forward_submodel(dummy, self.encoder) if self._use_encoder else dummy
            if self._is_discrete:
                a = T.randint(0, self.action_dim[0], (32,), device=self.device)
                a_in = T.nn.functional.one_hot(
                    a.long(), num_classes=int(np.prod(self.action_dim))).float()
            else:
                a_in = T.randn(32, *self.action_dim, device=self.device)
            self._forward_submodel(T.cat([s, ns], dim=-1), self.inverse_model)
            self._forward_submodel(T.cat([s, a_in], dim=-1), self.forward_model)

    def _embed(self, state: T.Tensor) -> T.Tensor:
        if self._use_encoder:
            return self._forward_submodel(state, self.encoder)
        return state

    def _full_forward(self, states, next_states, actions):
        encoded_s = self._embed(states)
        encoded_ns = self._embed(next_states)
        pred_a = self._forward_submodel(
            T.cat([encoded_s, encoded_ns], dim=-1), self.inverse_model)
        if self._is_discrete:
            actions_in = T.nn.functional.one_hot(
                actions.long().view(-1),
                num_classes=int(np.prod(self.action_dim))).float()
        else:
            actions_in = actions.to(self.device).float()
        pred_ns = self._forward_submodel(
            T.cat([encoded_s, actions_in], dim=-1), self.forward_model)
        return pred_a, pred_ns, encoded_ns

    def compute_learn_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        **kwargs: Any,
    ) -> T.Tensor:
        """
        Compute learn reward for a batch of states, next states, and actions.

        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).

        Returns:
            Tensor of shape (batch,) containing the learn reward for each state.
        """
        with T.no_grad():
            _, pred_ns, encoded_ns = self._full_forward(states, next_states, actions)
            err = (pred_ns - encoded_ns).pow(2).sum(dim=-1)

            # Feed error to reward normalizer, update, and normalize error
            self._feed_reward_normalizer(err)
            err = self._normalize_reward(err)

            # Return scaled intrinsic reward
            return 0.5 * self._scaled_reward_weight() * err

    def train(self, states, next_states, actions=None) -> T.Tensor:
        # Set mode of models and normalizers to train
        if self._use_encoder:
            self.encoder.train()
        self.inverse_model.train()
        self.forward_model.train()
        self.set_normalizers_mode('train')

        self.optimizer.zero_grad()
        pred_a, pred_ns, encoded_ns = self._full_forward(states, next_states, actions)

        if self._is_discrete:
            inverse_loss = T.nn.CrossEntropyLoss()(pred_a, actions.long().view(-1))
        else:
            inverse_loss = T.nn.MSELoss()(pred_a, actions.to(self.device).float())

        forward_loss = 0.5 * T.nn.MSELoss()(pred_ns, encoded_ns.detach())
        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        loss.backward()
        self.optimizer.step()

        # Set mode of models and normalizers to eval
        if self._use_encoder:
            self.encoder.eval()
        self.inverse_model.eval()
        self.forward_model.eval()
        self.set_normalizers_mode('eval')
        return loss

    def get_config(self) -> dict:
        return {
            'env': self.env.to_json(),
            'model_configs': self.model_configs,
            'optimizer_params': self.optimizer_params,
            'reward_weight': self.reward_weight,
            'reward_scheduler': (self.reward_scheduler.get_config()
                                 if self.reward_scheduler else None),
            'beta': self.beta,
            'extrinsic_threshold': self.extrinsic_threshold,
            'reward_normalizer': (
                self.reward_normalizer.get_config()
                if self.reward_normalizer else None)
        }

    @classmethod
    def _load_impl(cls, folder, config, env):
        from .normalizer import RewardNorm
        model_dir = Path(folder) / 'intrinsic_motivation'
        env_wrapper = env if env is not None else EnvWrapper.from_json(config['env'])

        sched = ScheduleWrapper(**config['reward_scheduler']) if config.get('reward_scheduler') else None

        ir_norm = None
        if config.get('reward_normalizer'):
            ir_norm = RewardNorm.load(
                config['reward_normalizer']['config'],
                str(model_dir / 'reward_normalizer_state.pt'))

        model = cls(
            env=env_wrapper,
            model_configs=config['model_configs'],
            optimizer_params=config['optimizer_params'],
            reward_weight=config['reward_weight'],
            reward_scheduler=sched,
            beta=config['beta'],
            extrinsic_threshold=config['extrinsic_threshold'],
            reward_normalizer=ir_norm,
            log_level=config.get('log_level', 'INFO'),
            device=config.get('device', None),
        )
        model.load_state_dict(T.load(model_dir / 'pytorch_model.pt'))
        return model

@register_intrinsic_motivation
class RND(IntrinsicMotivation):
    """
    Random Network Distillation (Burda et al. 2018).

    Two networks of identical architecture:
        target     f̂_φ : s -> R^k    (random init, FROZEN forever)
        predictor  f_θ : s -> R^k    (trained to match target on visited states)

    Intrinsic reward (per state s):
        r_i(s) = || f_θ(s) - f̂_φ(s) ||²

    Predictor training loss is the same quantity averaged over a minibatch.
    Both networks see normalized observations — this is essential. Without
    obs normalization the random target tends to collapse to a near-constant
    function on the data distribution and the predictor learns nothing.

    `model_configs` has two keys mirroring ICM's structure:
        {'target':    {'layer_config': [...], 'output_layer': [...]},
         'predictor': {'layer_config': [...], 'output_layer': [...]}}
    The two configs should match in shape (same output dim) but the predictor
    is allowed to be deeper / wider — the canonical setup uses identical nets.
    Default output dim if 'units' is absent: 512 (the NGU/RND default).
    """

    DEFAULT_OUTPUT_DIM = 512

    def __init__(
        self,
        env: EnvWrapper,
        model_configs: dict,
        optimizer_params: dict,
        reward_weight: float = 1.0,
        reward_scheduler: ScheduleWrapper | None = None,
        extrinsic_threshold: int = 0,
        reward_normalizer: RewardNorm | None = None,
        log_level: str = 'info',
        device: str | T.device | None = None,
    ):
        try:
            super().__init__(
                env=env,
                optimizer_params=optimizer_params,
                reward_weight=reward_weight,
                reward_scheduler=reward_scheduler,
                extrinsic_threshold=extrinsic_threshold,
                reward_normalizer=reward_normalizer,
                log_level=log_level,
                device=device,
            )
            self.model_configs = model_configs

            self.target: T.nn.ModuleDict | None = None
            self.predictor: T.nn.ModuleDict | None = None

            self._init_model()

            # Initialize predictor weights per config; target uses default random
            # init which is what we want — that random init *is* the function
            # we're trying to match.
            pred_cfg = (self.model_configs['predictor']['layer_config']
                        + self.model_configs['predictor']['output_layer'])
            self._init_weights(pred_cfg, self.predictor)

            # Freeze the target network forever
            for p in self.target.parameters():
                p.requires_grad = False

            # Optimizer covers only predictor params
            self.optimizer = self._init_optimizer(self.predictor.parameters())
            self.to(self.device)

        except Exception as e:
            self.logger.error(f"Error in RND init: {e}", exc_info=True)
            raise

    def _init_model(self) -> None:
        for name in ('target', 'predictor'):
            cfg = self.model_configs[name]
            module = T.nn.ModuleDict()
            for i, layer_info in enumerate(cfg['layer_config']):
                layer = self._build_layer(layer_info['type'],
                                          layer_info.get('params', {}).copy())
                module[f"{name}_{layer_info['type']}_{i}"] = layer
            output_info = cfg['output_layer'][0]
            out_units = output_info.get('params', {}).get('units',
                                                          self.DEFAULT_OUTPUT_DIM)
            module[f"{name}_{output_info['type']}_output"] = T.nn.LazyLinear(out_units)
            module.to(self.device)
            setattr(self, name, module)
            self.add_module(name, module)

        # Infer LazyLinear shapes
        with T.no_grad():
            dummy = T.ones((32, *self.obs_dim), device=self.device, dtype=T.float)
            self._forward_submodel(dummy, self.target)
            self._forward_submodel(dummy, self.predictor)

    def _embed(self, x: T.Tensor):
        """Return (target_embedding, predictor_embedding) for a state batch."""
        with T.no_grad():
            t_out = self._forward_submodel(x, self.target)
        p_out = self._forward_submodel(x, self.predictor)
        return t_out, p_out

    def compute_learn_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        **kwargs: Any,
    ) -> T.Tensor:
        """
        Compute learn reward for a batch of states, next states, and actions.

        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).

        Returns:
            Tensor of shape (batch,) containing the learn reward for each state.
        """
        with T.no_grad():
            t_out, p_out = self._embed(next_states)
            err = (p_out - t_out).pow(2).sum(dim=-1)

            # Feed error to reward normalizer, update, and normalize error
            if self.reward_normalizer is not None:
                dones = T.zeros(err.shape, dtype=T.bool, device=self.device)
                self.reward_normalizer.add(err, dones)
                self.reward_normalizer.update()
                err = self.reward_normalizer.normalize(err)
            
            # Return scaled intrinsic reward
            return self._scaled_reward_weight() * err

    def train(self, states, next_states, actions=None) -> T.Tensor:
        # Set mode of models and normalizers to train
        self.predictor.train()
        self.set_normalizers_mode('train')

        self.optimizer.zero_grad()
        t_out, p_out = self._embed(next_states)
        # The loss is exactly the intrinsic-reward formula (averaged)
        loss = (p_out - t_out.detach()).pow(2).sum(dim=-1).mean()
        loss.backward()
        self.optimizer.step()

        # Set mode of models and normalizers to eval
        self.predictor.eval()
        self.set_normalizers_mode('eval')

        return loss

    # ----------------------------- save / load
    def get_config(self) -> dict:
        return {
            'env': self.env.to_json(),
            'model_configs': self.model_configs,
            'optimizer_params': self.optimizer_params,
            'reward_weight': self.reward_weight,
            'reward_scheduler': (self.reward_scheduler.get_config()
                                 if self.reward_scheduler else None),
            'extrinsic_threshold': self.extrinsic_threshold,
            'reward_normalizer': (
                self.reward_normalizer.get_config()
                if self.reward_normalizer else None)
        }

    @classmethod
    def _load_impl(cls, folder, config, env):
        from .normalizer import RewardNorm
        model_dir = Path(folder) / 'intrinsic_motivation'
        env_wrapper = env if env is not None else EnvWrapper.from_json(config['env'])

        sched = ScheduleWrapper(**config['reward_scheduler']) if config.get('reward_scheduler') else None

        ir_norm = None
        if config.get('reward_normalizer'):
            ir_norm = RewardNorm.load(
                config['reward_normalizer']['config'],
                str(model_dir / 'reward_normalizer_state.pt'))

        model = cls(
            env=env_wrapper,
            model_configs=config['model_configs'],
            optimizer_params=config['optimizer_params'],
            reward_weight=config['reward_weight'],
            reward_scheduler=sched,
            extrinsic_threshold=config['extrinsic_threshold'],
            reward_normalizer=ir_norm,
            log_level=config.get('log_level', 'INFO'),
            device=config.get('device', None),
        )
        model.load_state_dict(T.load(model_dir / 'pytorch_model.pt'))
        return model

@register_intrinsic_motivation
class EpisodicNovelty(IntrinsicMotivation):
    """
    Episodic novelty signal from NGU (Badia et al. 2020).

    Per-env episodic memory M_i (one per parallel env). At each step we:
      1. Embed s_{t+1} via an encoder φ trained by inverse dynamics
         (so φ encodes only controllable features — same trick as ICM).
      2. Query M_i for the k nearest neighbors of φ(s_{t+1}).
      3. Compute the novelty bonus from kernel-summed inverse distances:

            α_epi = 1 / sqrt( Σ_{f_j ∈ N_k}  K(φ(s_{t+1}), f_j)  +  c )

         where K(x, y) = ε / ( d(x,y)² / d̄² + ε )
         and d̄ is a running mean of squared distances (paper-default).

      4. Append φ(s_{t+1}) to M_i.

    On episode end, M_i is cleared for env i.

    Training: encoder φ is trained jointly with an inverse dynamics head
    g(φ(s), φ(s')) -> â, identical to ICM's encoder/inverse pair.
    """

    def __init__(
        self,
        env: EnvWrapper,
        model_configs: dict,
        optimizer_params: dict,
        memory_size: int = 30_000,
        k: int = 10,
        kernel_epsilon: float = 1e-3,
        cluster_distance: float = 8e-3,   # `c` in the formula
        max_similarity: float = 8.0,
        running_mean_decay: float = 0.99,
        reward_weight: float = 1.0,
        reward_scheduler: ScheduleWrapper | None = None,
        extrinsic_threshold: int = 0,
        reward_normalizer: RewardNorm | None = None,
        log_level: str = 'info',
        device: str | T.device | None = None,
    ):
        try:
            super().__init__(
                env=env,
                optimizer_params=optimizer_params,
                reward_weight=reward_weight,
                reward_scheduler=reward_scheduler,
                extrinsic_threshold=extrinsic_threshold,
                reward_normalizer=reward_normalizer,
                log_level=log_level,
                device=device,
            )
            self.is_online = True
            self.model_configs = model_configs
            self.memory_size = memory_size
            self.k = k
            self.kernel_epsilon = kernel_epsilon
            self.cluster_distance = cluster_distance
            self.max_similarity = max_similarity
            self.running_mean_decay = running_mean_decay

            # Networks
            self.encoder: T.nn.ModuleDict | None = None
            self.inverse_model: T.nn.ModuleDict | None = None
            self._init_model()

            enc_cfg = (self.model_configs['encoder']['layer_config']
                       + self.model_configs['encoder']['output_layer'])
            self._init_weights(enc_cfg, self.encoder)
            inv_cfg = (self.model_configs['inverse_model']['layer_config']
                       + self.model_configs['inverse_model']['output_layer'])
            self._init_weights(inv_cfg, self.inverse_model)

            self.optimizer = self._init_optimizer()

            # Per-env episodic memory: list of (memory_size, embed_dim) tensors
            num_envs = getattr(env, 'num_envs', 1)
            self.num_envs = num_envs
            embed_dim = self.model_configs['encoder']['output_layer'][0]\
                .get('params', {}).get('units', 32)
            self.embed_dim = embed_dim
            # Stored on CPU to keep GPU memory free; moved to device during query
            self._memories: List[T.Tensor] = [
                T.zeros((0, embed_dim), dtype=T.float32) for _ in range(num_envs)
            ]
            # Running mean of squared distances (used to scale the kernel)
            self._running_sq_dist = T.tensor(1.0, device=self.device)

            self.to(self.device)

        except Exception as e:
            self.logger.error(f"Error in EpisodicNovelty init: {e}", exc_info=True)
            raise

    def _init_model(self) -> None:
        for name in ('encoder', 'inverse_model'):
            cfg = self.model_configs[name]
            module = T.nn.ModuleDict()
            for i, layer_info in enumerate(cfg['layer_config']):
                layer = self._build_layer(layer_info['type'],
                                          layer_info.get('params', {}).copy())
                module[f"{name}_{layer_info['type']}_{i}"] = layer
            output_info = cfg['output_layer'][0]
            if name == 'inverse_model':
                out_units = output_info.get('params', {}).get(
                    'units', int(np.prod(self.action_dim)))
            else:
                out_units = output_info.get('params', {}).get('units', 32)
            module[f"{name}_{output_info['type']}_output"] = T.nn.LazyLinear(out_units)
            module.to(self.device)
            setattr(self, name, module)
            self.add_module(name, module)

        # Materialize LazyLinear shapes
        with T.no_grad():
            dummy = T.ones((32, *self.obs_dim), device=self.device, dtype=T.float)
            phi = self._forward_submodel(dummy, self.encoder)
            self._forward_submodel(T.cat([phi, phi], dim=-1), self.inverse_model)

    def _embed(self, x: T.Tensor) -> T.Tensor:
        return self._forward_submodel(x, self.encoder)

    def _knn_bonus(self, embeddings: T.Tensor, env_indices: T.Tensor) -> T.Tensor:
        """
        For each row i, query memory[env_indices[i]] for k-NN of embeddings[i]
        and return the per-row bonus α_epi.
        """
        bonuses = T.zeros(embeddings.shape[0], device=self.device)
        for i in range(embeddings.shape[0]):
            env_i = int(env_indices[i].item()) if env_indices is not None else 0
            mem = self._memories[env_i].to(self.device)
            if mem.shape[0] < 2:
                # No useful memory yet — give a neutral bonus of 1.0
                bonuses[i] = 1.0
                continue
            x = embeddings[i].unsqueeze(0)                  # (1, d)
            d2 = ((mem - x) ** 2).sum(dim=-1)               # (M,)
            k = min(self.k, d2.shape[0])
            topk_d2, _ = d2.topk(k, largest=False)          # smallest k
            # Update running mean of squared distance
            self._running_sq_dist = (
                self.running_mean_decay * self._running_sq_dist
                + (1 - self.running_mean_decay) * topk_d2.mean()
            )
            # Inverse-distance kernel, normalized by running mean
            d2n = topk_d2 / (self._running_sq_dist + 1e-8)
            kernel = self.kernel_epsilon / (d2n + self.kernel_epsilon)
            sim = kernel.sum()
            # Cap and convert to bonus
            if sim > self.max_similarity ** 2:
                bonuses[i] = 0.0
            else:
                bonuses[i] = 1.0 / T.sqrt(sim + self.cluster_distance)
        return bonuses

    def _append_to_memory(self, embeddings: T.Tensor, env_indices: T.Tensor) -> None:
        emb_cpu = embeddings.detach().cpu()
        for i in range(emb_cpu.shape[0]):
            env_i = int(env_indices[i].item()) if env_indices is not None else 0
            self._memories[env_i] = T.cat([self._memories[env_i],
                                           emb_cpu[i:i+1]], dim=0)
            if self._memories[env_i].shape[0] > self.memory_size:
                # Drop oldest
                self._memories[env_i] = self._memories[env_i][-self.memory_size:]

    def compute_rollout_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        env_indices: T.Tensor | None = None,
    ) -> T.Tensor:
        """
        Compute rollout reward for a batch of states, next states, and actions.

        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).
            env_indices: Tensor of shape (batch,) containing the environment indices.

        Returns:
            Tensor of shape (batch,) containing the rollout reward for each state.
        """
        with T.no_grad():
            if env_indices is None:
                env_indices = T.arange(next_states.shape[0], dtype=T.long)
            embeddings = self._embed(next_states)
            bonus = self._knn_bonus(embeddings, env_indices)
            
            # Feed error to reward normalizer, update, and normalize error
            if self.reward_normalizer is not None:
                dones = T.zeros(bonus.shape, dtype=T.bool, device=self.device)
                self.reward_normalizer.add(bonus, dones)
                self.reward_normalizer.update()
                bonus = self.reward_normalizer.normalize(bonus)
            
            self._append_to_memory(embeddings, env_indices)
            return self._scaled_reward_weight() * bonus

    def train(self, states, next_states, actions=None) -> T.Tensor:
        self.encoder.train()
        self.inverse_model.train()
        self.optimizer.zero_grad()

        phi_s = self._embed(states)
        phi_ns = self._embed(next_states)
        pred_a = self._forward_submodel(T.cat([phi_s, phi_ns], dim=-1),
                                        self.inverse_model)
        if self._is_discrete:
            loss = T.nn.CrossEntropyLoss()(pred_a, actions.long().view(-1))
        else:
            loss = T.nn.MSELoss()(pred_a, actions.to(self.device).float())

        loss.backward()
        self.optimizer.step()

        self.encoder.eval()
        self.inverse_model.eval()
        return loss

    def on_episode_end(self, env_indices: T.Tensor) -> None:
        """Clear episodic memory for envs that just finished."""
        for i in env_indices.flatten().tolist():
            self._memories[int(i)] = T.zeros((0, self.embed_dim), dtype=T.float32)

    def get_config(self) -> dict:
        return {
            'env': self.env.to_json(),
            'model_configs': self.model_configs,
            'optimizer_params': self.optimizer_params,
            'memory_size': self.memory_size,
            'k': self.k,
            'kernel_epsilon': self.kernel_epsilon,
            'cluster_distance': self.cluster_distance,
            'max_similarity': self.max_similarity,
            'running_mean_decay': self.running_mean_decay,
            'reward_weight': self.reward_weight,
            'reward_scheduler': (self.reward_scheduler.get_config()
                                 if self.reward_scheduler else None),
            'extrinsic_threshold': self.extrinsic_threshold,
            'reward_normalizer': (
                self.reward_normalizer.get_config()
                if self.reward_normalizer else None)
        }

    @classmethod
    def _load_impl(cls, folder, config, env):
        from .normalizer import RewardNorm
        model_dir = Path(folder) / 'intrinsic_motivation'
        env_wrapper = env if env is not None else EnvWrapper.from_json(config['env'])

        sched = ScheduleWrapper(**config['reward_scheduler']) if config.get('reward_scheduler') else None

        ir_norm = None
        if config.get('reward_normalizer'):
            ir_norm = RewardNorm.load(
                config['reward_normalizer']['config'],
                str(model_dir / 'reward_normalizer_state.pt'))

        model = cls(
            env=env_wrapper,
            model_configs=config['model_configs'],
            optimizer_params=config['optimizer_params'],
            memory_size=config['memory_size'],
            k=config['k'],
            kernel_epsilon=config['kernel_epsilon'],
            cluster_distance=config['cluster_distance'],
            max_similarity=config['max_similarity'],
            running_mean_decay=config['running_mean_decay'],
            reward_weight=config['reward_weight'],
            reward_scheduler=sched,
            extrinsic_threshold=config['extrinsic_threshold'],
            reward_normalizer=ir_norm,
            log_level=config.get('log_level', 'INFO'),
            device=config.get('device', None),
        )
        model.load_state_dict(T.load(model_dir / 'pytorch_model.pt'))
        return model


def additive_combination(rewards: List[T.Tensor],
                         weights: Sequence[float] | None = None) -> T.Tensor:
    """r = Σ_i w_i * r_i.  weights default to 1.0 each."""
    if not rewards:
        raise ValueError("Empty rewards list")
    if weights is None:
        weights = [1.0] * len(rewards)
    out = weights[0] * rewards[0]
    for w, r in zip(weights[1:], rewards[1:]):
        out = out + w * r
    return out


def multiplicative_combination(rewards: List[T.Tensor]) -> T.Tensor:
    """r = ∏_i r_i.  Useful when you want every signal to fire."""
    if not rewards:
        raise ValueError("Empty rewards list")
    out = rewards[0]
    for r in rewards[1:]:
        out = out * r
    return out


def max_combination(rewards: List[T.Tensor]) -> T.Tensor:
    """r = max_i r_i.  Strongest single signal wins per step."""
    if not rewards:
        raise ValueError("Empty rewards list")
    stacked = T.stack(rewards, dim=0)
    return stacked.max(dim=0).values


def ngu_combination(rewards: List[T.Tensor], L: float = 5.0) -> T.Tensor:
    """
    NGU's reward: r = α_epi * clip(α_lifelong, 1, L)
    Expects rewards in order: [episodic, lifelong].
    """
    if len(rewards) != 2:
        raise ValueError("ngu_combination expects exactly 2 rewards: [episodic, lifelong]")
    episodic, lifelong = rewards
    return episodic * lifelong.clamp(min=1.0, max=L)


# Named registry so configs can store the rule by string
_COMBINATION_RULES: dict[str, Callable] = {
    'additive': additive_combination,
    'multiplicative': multiplicative_combination,
    'max': max_combination,
    'ngu': ngu_combination,
}


@register_intrinsic_motivation
class CompositeIntrinsicMotivation(IntrinsicMotivation):
    """
    Wraps a list of IntrinsicMotivation modules and combines their per-step
    intrinsic rewards according to a combination rule.

    The composite has no networks of its own and no optimizer — every child
    manages its own training. Its `train()` calls each child's `train()` and
    returns the sum of losses (for logging only).
    """

    def __init__(
        self,
        env: EnvWrapper,
        components: List[IntrinsicMotivation],
        combination_rule: str = 'additive',
        combination_kwargs: dict | None = None,
        reward_weight: float = 1.0,
        reward_scheduler: ScheduleWrapper | None = None,
        extrinsic_threshold: int = 0,
        log_level: str = 'info',
        device: str | T.device | None = None,
    ):
        try:
            super().__init__(
                env=env,
                optimizer_params=None,
                reward_weight=reward_weight,
                reward_scheduler=reward_scheduler,
                extrinsic_threshold=extrinsic_threshold,
                reward_normalizer=None,
                log_level=log_level,
                device=device,
            )
            if combination_rule not in _COMBINATION_RULES:
                raise ValueError(f"Unknown combination_rule: {combination_rule}")
            self.combination_rule = combination_rule
            self.combination_kwargs = combination_kwargs or {}
            self.components: List[IntrinsicMotivation] = components

            # Register components so .to(device), .state_dict() etc. propagate
            for i, c in enumerate(components):
                self.add_module(f"component_{i}", c)

            self.to(self.device)
        except Exception as e:
            self.logger.error(f"Error in CompositeIntrinsicMotivation init: {e}",
                              exc_info=True)
            raise

    def _split_components(self) -> tuple[list[IntrinsicMotivation], list[IntrinsicMotivation]]:
        """
        Splits components into online and parametric components.

        Returns:
            tuple of lists: (online components, parametric components)
        """
        online, parametric = [], []
        for i, c in enumerate(self.components):
            if c.is_online:
                online.append((i, c))
            else:
                parametric.append((i, c))
        return online, parametric

    def _weights_for(self, components: list[tuple[int, IntrinsicMotivation]]) -> list[float]:
        """
        Returns weights for list of components.

        Args:
            components: List of tuples: (index, component)

        Returns:
            List of weights for each component.
        """
        weights = self.combination_kwargs.get('weights')
        if weights is None:
            return None
        return [weights[i] for i, _ in components]

    def compute_rollout_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        env_indices: T.Tensor | None = None,
    ) -> T.Tensor:
        """
        Compute rollout intrinsic reward for a batch of states, next state, and actions
        across all intrinsic motivations in the composite.

        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).
            env_indices: Tensor of shape (batch,) containing the environment indices.

        Returns:
            Tensor of shape (batch,) containing the rollout reward for each state.
        """
        rule = _COMBINATION_RULES[self.combination_rule]
        online, _ = self._split_components()
        if not online:
            return T.zeros(states.shape[0], device=self.device, dtype=T.float32)

        rewards = [c.compute_rollout_reward(states, next_states, actions, env_indices) for _, c in online]
        weights = self._weights_for(online)

        if self.combination_rule == 'additive':
            return rule(rewards, weights=weights)
        return rule(rewards)

    def compute_learn_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        rollout_rewards: T.Tensor | None = None,
    ) -> T.Tensor:
        """
        Compute learn reward for a batch of states, next states, and actions
        across all intrinsic motivations in the composite and adds to them the rollout rewards
        if provided.

        Args:
            states: Tensor of shape (batch, state_dim).
            next_states: Tensor of shape (batch, state_dim).
            actions: Tensor of shape (batch, action_dim).
            rollout_rewards: Tensor of shape (batch,) containing the rollout reward for each state.

        Returns:
            Tensor of shape (batch,) containing the learn reward for each state.
        """
        rule = _COMBINATION_RULES[self.combination_rule]
        online, parametric = self._split_components()

        if parametric:
            learn_rewards = [c.compute_learn_reward(states, next_states, actions) for _, c in parametric]
            learn_weights = self._weights_for(parametric)
            learn_combined = (rule(learn_rewards, weights=learn_weights)
                              if self.combination_rule == 'additive'
                              else rule(learn_rewards))
        else:
            learn_combined = None

        rollout_combined = rollout_rewards if online else None

        if learn_combined is None and rollout_combined is None:
            raise RuntimeError("No learn or rollout rewards to combine")
        if learn_combined is None:
            final_rewards = rollout_combined
        elif rollout_combined is None:
            final_rewards = learn_combined
        else:
            final_rewards = rule([rollout_combined, learn_combined])

        return self._scaled_reward_weight() * final_rewards

    def train(self, states, next_states, actions=None) -> T.Tensor:
        total = T.tensor(0.0, device=self.device)
        for c in self.components:
            loss = c.train(states, next_states, actions)
            total = total + loss.detach()
        return total

    def on_episode_end(self, env_indices: T.Tensor) -> None:
        for c in self.components:
            c.on_episode_end(env_indices)

    def set_normalizers_mode(self, context: Literal['train', 'test']) -> None:
        for c in self.components:
            c.set_normalizers_mode(context)

    # @property
    # def is_online(self) -> bool:
    #     return any(c.is_online for c in self.components)

    def save(self, folder) -> None:
        comp_root = Path(folder) / 'intrinsic_motivation'
        comp_root.mkdir(parents=True, exist_ok=True)
        # Save each component to its own subdirectory
        component_dirs = []
        for i, c in enumerate(self.components):
            sub = comp_root / f'component_{i}_{c.__class__.__name__}'
            sub.mkdir(parents=True, exist_ok=True)
            # Each component's save expects a parent folder; we pass `sub` so it
            # creates `sub/intrinsic_motivation/...`
            c.save(sub)
            component_dirs.append(sub.name)

        config = self.get_config()
        config['type'] = self.__class__.__name__
        config['component_dirs'] = component_dirs
        with open(comp_root / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f)

    def get_config(self) -> dict:
        return {
            'env': self.env.to_json(),
            'components': [c.get_config() for c in self.components],
            'combination_rule': self.combination_rule,
            'combination_kwargs': self.combination_kwargs,
            'reward_weight': self.reward_weight,
            'reward_scheduler': (self.reward_scheduler.get_config()
                                 if self.reward_scheduler else None),
            'extrinsic_threshold': self.extrinsic_threshold
        }

    @classmethod
    def _load_impl(cls, folder, config, env):
        comp_root = Path(folder) / 'intrinsic_motivation'

        sched = (ScheduleWrapper(**config['im_reward_scheduler'])
                 if config.get('im_reward_scheduler') else None)

        # Load each component recursively via the base `load` dispatcher
        components = []
        for sub_name in config['component_dirs']:
            sub = comp_root / sub_name
            components.append(IntrinsicMotivation.load(sub, env=env))

        env_wrapper = env if env is not None else EnvWrapper.from_json(config['env'])
        return cls(
            env=env_wrapper,
            components=components,
            combination_rule=config['combination_rule'],
            combination_kwargs=config.get('combination_kwargs', {}),
            reward_weight=config['reward_weight'],
            reward_scheduler=sched,
            extrinsic_threshold=config['extrinsic_threshold'],
            log_level=config.get('log_level', 'INFO'),
            device=config.get('device', None),
        )