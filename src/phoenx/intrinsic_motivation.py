"""Pluggable intrinsic-motivation modules for exploration bonuses.

Concrete subclasses sit under ``IntrinsicMotivation`` (itself a ``Model``):

- ``ICM`` — Intrinsic Curiosity Module (Pathak et al., 2017): forward-model
  prediction error in a learned feature space, shaped by an inverse-dynamics
  head so features encode controllable aspects of the observation.
- ``RND`` — Random Network Distillation (Burda et al., 2018): squared error
  between a frozen random target network and a trained predictor.
- ``EpisodicNovelty`` — the k-NN episodic component of NGU (Badia et al.,
  2020): per-env memory of embeddings with a kernel-summed inverse-distance
  bonus, cleared on episode end.
- ``CompositeIntrinsicMotivation`` — wraps several modules and merges their
  per-step rewards with a named combination rule.

Trainer integration is additive. Modules expose ``compute_learn_reward`` /
``compute_rollout_reward`` (batch intrinsic bonuses), ``train`` (one update
step), ``on_episode_end`` (stateful hooks such as episodic memory reset),
``set_normalizers_mode``, and ``save`` / ``load``. Shared knobs include
``reward_weight``, an optional ``reward_scheduler``, and
``extrinsic_threshold`` for intrinsic-only warmup.

Combination rules used by ``CompositeIntrinsicMotivation`` (selected by
string name): ``additive_combination``, ``multiplicative_combination``,
``max_combination``, and ``ngu_combination``.
"""

from abc import abstractmethod
from typing import Callable, List, Literal, Sequence, Any
from pathlib import Path
import json

import torch as T
import numpy as np
import gymnasium as gym

from .models import Model
from .env_wrapper import EnvWrapper
from .schedulers import ScheduleWrapper
from .normalizer import RewardNorm
from .logging_config import get_logger


# =============================================================================
# Type registry — used by IntrinsicMotivation.load() to dispatch to subclasses
# =============================================================================
_REGISTRY: dict[str, type] = {}


def register_intrinsic_motivation(cls):
    """Register an ``IntrinsicMotivation`` subclass for save/load dispatch.

    Args:
        cls (type): Subclass to register under ``cls.__name__``.

    Returns:
        registered (type): The same class, unchanged, so the decorator can
            wrap a class body.
    """
    _REGISTRY[cls.__name__] = cls
    return cls

class IntrinsicMotivation(Model):
    """Abstract base for modules that produce intrinsic rewards.

    Inherits from ``Model`` so subclasses reuse ``_build_layer``,
    ``_init_weights``, ``_init_optimizer``, and device handling. The base
    itself builds no networks — it strips the empty ``layers`` ModuleDict
    that ``Model`` creates and provides shared bookkeeping.

    Attributes:
        reward_weight: Scalar multiplier on the intrinsic reward.
        reward_scheduler: Optional schedule that scales ``reward_weight``.
        extrinsic_threshold: Agent step below which extrinsic reward is
            suppressed (intrinsic-only warmup).
        reward_normalizer: Optional ``RewardNorm`` over intrinsic rewards.
        is_online: When True, the module expects per-step rollout updates
            (e.g. episodic memory) rather than only learn-time bonuses.
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
        """Configure shared intrinsic-motivation bookkeeping.

        Args:
            env: Environment wrapper used for observation/action shapes.
            optimizer_params: Optimizer config forwarded to ``Model``; may be
                ``None`` when the subclass builds no optimizer.
            reward_weight: Base scale on intrinsic rewards.
            reward_scheduler: Optional decay/schedule applied to
                ``reward_weight``.
            extrinsic_threshold: Steps of intrinsic-only warmup before
                extrinsic reward is restored.
            reward_normalizer: Optional running normalizer for intrinsic
                rewards.
            log_level: Logger level name (e.g. ``'info'``).
            device: Torch device, or ``None`` for the framework default.
        """
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
        """Run one training step and return a scalar loss.

        Args:
            states (torch.Tensor): Batch of states.
            next_states (torch.Tensor): Batch of next states.
            actions (torch.Tensor | None): Batch of actions when the module
                conditions on them.

        Returns:
            Scalar loss tensor.
        """
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """Return a JSON-serializable config excluding tensor state."""
        ...
        
    def use_extrinsic_reward(self, step: int) -> bool:
        """Return whether extrinsic reward should be used at ``step``."""
        return step >= self.extrinsic_threshold

    def compute_learn_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        **kwargs: Any,
    ) -> T.Tensor:
        """Return learn-time intrinsic reward of shape ``(batch,)``.

        The base implementation returns zeros; subclasses override.

        Args:
            states: Batch of states, shape ``(batch, *state_dim)``.
            next_states: Batch of next states, shape ``(batch, *state_dim)``.
            actions: Batch of actions, shape ``(batch, *action_dim)``, or
                ``None`` when unused.
            **kwargs: Extra keyword arguments forwarded by callers; ignored
                by the base implementation.

        Returns:
            Per-sample learn reward of shape ``(batch,)``.
        """
        return T.zeros(states.shape[0], device=self.device, dtype=T.float32)

    def compute_rollout_reward(
        self,
        states: T.Tensor,
        next_states: T.Tensor,
        actions: T.Tensor | None = None,
        env_indices: T.Tensor | None = None,
    ) -> T.Tensor:
        """Return rollout-time intrinsic reward of shape ``(batch,)``.

        The base implementation returns zeros; subclasses override.

        Args:
            states: Batch of states, shape ``(batch, *state_dim)``.
            next_states: Batch of next states, shape ``(batch, *state_dim)``.
            actions: Batch of actions, shape ``(batch, *action_dim)``, or
                ``None`` when unused.
            env_indices: Parallel-env indices of shape ``(batch,)``, or
                ``None`` when a single env is assumed.

        Returns:
            Per-sample rollout reward of shape ``(batch,)``.
        """
        return T.zeros(states.shape[0], device=self.device, dtype=T.float32)

    def on_episode_end(self, env_indices: T.Tensor) -> None:
        """Handle episode boundaries for the given parallel env indices.

        Called by the Trainer for each parallel env that just terminated or
        truncated. The base implementation is a no-op.

        Args:
            env_indices: 1-D long tensor of env indices that finished.
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
        """Set attached reward normalizers to train or eval mode.

        Args:
            context: ``'train'`` or ``'eval'``.
        """
        for n in [self.reward_normalizer]:
            if n is None:
                continue
            n.train() if context == 'train' else n.eval()

    def _feed_reward_normalizer(self, reward: T.Tensor) -> None:
        """Feed a reward batch into ``reward_normalizer`` and update it.

        Args:
            reward: Reward tensor to feed.
        """
        if self.reward_normalizer is not None:
            dones = T.zeros(reward.shape, dtype=T.bool, device=self.device)
            self.reward_normalizer.add(reward, dones)
            self.reward_normalizer.update()

    def _normalize_reward(self, intrinsic_reward: T.Tensor) -> T.Tensor:
        """Normalize an intrinsic reward through ``reward_normalizer`` if set.

        Args:
            intrinsic_reward: Reward tensor to normalize.

        Returns:
            Normalized tensor, or the input unchanged when no normalizer is
            present.
        """
        if self.reward_normalizer is not None:
            return self.reward_normalizer.normalize(intrinsic_reward)
        return intrinsic_reward

    def _scaled_reward_weight(self) -> float:
        """Return ``reward_weight`` scaled by the optional reward scheduler."""
        w = self.reward_weight
        if self.reward_scheduler is not None:
            w *= self.reward_scheduler.get_factor()
        return w

    def _forward_submodel(self, x: T.Tensor, submodel: T.nn.ModuleDict) -> T.Tensor:
        """Run ``x`` sequentially through every layer in ``submodel``.

        Args:
            x: Input tensor.
            submodel: Ordered ``ModuleDict`` of layers.

        Returns:
            Output of the last layer.
        """
        for _, layer in submodel.items():
            x = layer(x)
        return x

    def save(self, folder) -> None:
        """Persist weights, config, and optional normalizer state under ``folder``.

        Writes ``folder/intrinsic_motivation/pytorch_model.pt`` and
        ``config.json``, tagging ``config['type']`` with the class name so
        ``load`` can dispatch. When a reward normalizer is present its state
        is saved alongside.

        Args:
            folder (str | pathlib.Path): Parent directory for the save tree.
        """
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
        """Construct a concrete intrinsic-motivation module by type name.

        Args:
            im_type: One of ``'ICM'``, ``'RND'``, or ``'EpisodicNovelty'``.
            **kwargs (Any): Forwarded to the subclass constructor.

        Returns:
            A new ``ICM``, ``RND``, or ``EpisodicNovelty`` instance.

        Raises:
            ValueError: If ``im_type`` is not a known concrete type.
        """
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
        """Load a saved module, dispatching on the config ``type`` field.

        Args:
            folder: Parent directory that contains ``intrinsic_motivation/``.
            env: Optional live env to reuse; when ``None``, the env is rebuilt
                from the saved config.

        Returns:
            Restored subclass instance registered under the saved type name.

        Raises:
            FileNotFoundError: If ``config.json`` is missing.
            ValueError: If the saved type is not in the registry.
        """
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
        """Subclass hook that rebuilds an instance from a saved config.

        Args:
            folder (str | pathlib.Path): Parent save directory.
            config (dict): Parsed ``config.json`` contents.
            env (EnvWrapper | None): Optional live env to reuse.

        Returns:
            Restored instance of ``cls``.

        Raises:
            NotImplementedError: Always, until a subclass overrides this.
        """
        raise NotImplementedError(f"{cls.__name__} must implement _load_impl")

@register_intrinsic_motivation
class ICM(IntrinsicMotivation):
    """Intrinsic Curiosity Module (Pathak et al., 2017).

    Three optional/required sub-networks:

    - encoder ``φ``: maps observations to features (optional; raw obs used
      when absent);
    - inverse ``g``: predicts the action from ``(φ(s), φ(s'))``, shaping ``φ``
      toward controllable features;
    - forward ``f``: predicts ``φ(s')`` from ``(φ(s), a)``; squared error in
      feature space is the intrinsic reward.

    Training loss is ``(1 - β) * L_inv + β * L_fwd``. The intrinsic reward is
    ``r_i = (η/2) * ||f(φ(s), a) - φ(s')||²`` with ``η`` from
    ``reward_weight`` (and its scheduler).

    Attributes:
        model_configs: Layer configs for encoder / inverse / forward nets.
        beta: Mixing weight between inverse and forward losses.
        encoder: Optional feature encoder ``ModuleDict``.
        inverse_model: Inverse-dynamics ``ModuleDict``.
        forward_model: Forward-dynamics ``ModuleDict``.
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
        """Build encoder, inverse, and forward networks for ICM.

        Args:
            env: Environment wrapper used for observation/action shapes.
            model_configs: Mapping with keys ``'inverse_model'`` and
                ``'forward_model'``, and optionally ``'encoder'``. Each value
                has ``layer_config`` and ``output_layer`` lists.
            optimizer_params: Optimizer configuration for ICM parameters.
            reward_weight: Base scale on the forward-model error reward.
            reward_scheduler: Optional schedule applied to ``reward_weight``.
            beta: Weight of the forward loss versus the inverse loss.
            extrinsic_threshold: Intrinsic-only warmup steps.
            reward_normalizer: Optional normalizer over intrinsic rewards.
            log_level: Logger level name.
            device: Torch device, or ``None`` for the framework default.
        """
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
        """Build ICM submodules from ``model_configs`` and materialize LazyLinear."""
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
        """Encode ``state`` when an encoder is present; otherwise return it.

        Args:
            state: Observation batch.

        Returns:
            Feature batch (encoder output or raw observations).
        """
        if self._use_encoder:
            return self._forward_submodel(state, self.encoder)
        return state

    def _full_forward(self, states, next_states, actions):
        """Run inverse and forward heads for a transition batch.

        Args:
            states (torch.Tensor): Batch of states.
            next_states (torch.Tensor): Batch of next states.
            actions (torch.Tensor): Batch of actions.

        Returns:
            pred_a (torch.Tensor): Inverse-model action prediction.
            pred_ns (torch.Tensor): Forward-model next-feature prediction.
            encoded_ns (torch.Tensor): Embedded next states (detached target
                for the forward loss when training).
        """
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
        """Compute ICM learn reward from forward-model feature error.

        Args:
            states: Batch of states, shape ``(batch, *state_dim)``.
            next_states: Batch of next states, shape ``(batch, *state_dim)``.
            actions: Batch of actions, shape ``(batch, *action_dim)``.
            **kwargs: Extra keyword arguments ignored by ICM.

        Returns:
            Per-sample intrinsic reward of shape ``(batch,)``.
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
        """Update ICM networks on a transition batch.

        Args:
            states (torch.Tensor): Batch of states.
            next_states (torch.Tensor): Batch of next states.
            actions (torch.Tensor | None): Batch of actions.

        Returns:
            Scalar combined inverse/forward loss.
        """
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
        """Return a JSON-serializable ICM configuration."""
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
        """Rebuild an ICM instance from a saved config and weights.

        Args:
            folder (str | pathlib.Path): Parent save directory.
            config (dict): Parsed ``config.json`` contents.
            env (EnvWrapper | None): Optional live env to reuse.

        Returns:
            Restored ``ICM`` with loaded ``state_dict``.
        """
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
    """Random Network Distillation (Burda et al., 2018).

    Two networks of matching output dimension:

    - target ``f̂_φ``: randomly initialized and frozen forever;
    - predictor ``f_θ``: trained to match the target on visited states.

    Intrinsic reward per next state is
    ``r_i(s') = ||f_θ(s') - f̂_φ(s')||²``; training minimizes the same
    quantity averaged over a minibatch. Observation normalization is
    essential in practice — without it the frozen target tends to collapse
    toward a near-constant function.

    ``model_configs`` mirrors ICM's structure with keys ``'target'`` and
    ``'predictor'``, each holding ``layer_config`` and ``output_layer``.
    Default output dim when ``units`` is absent is ``DEFAULT_OUTPUT_DIM``
    (512, the NGU/RND default).

    Attributes:
        model_configs: Layer configs for target and predictor nets.
        target: Frozen random target ``ModuleDict``.
        predictor: Trainable predictor ``ModuleDict``.
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
        """Build frozen target and trainable predictor networks for RND.

        Args:
            env: Environment wrapper used for observation shapes.
            model_configs: Mapping with ``'target'`` and ``'predictor'``
                entries, each containing ``layer_config`` and
                ``output_layer``.
            optimizer_params: Optimizer configuration for predictor params.
            reward_weight: Base scale on the prediction-error reward.
            reward_scheduler: Optional schedule applied to ``reward_weight``.
            extrinsic_threshold: Intrinsic-only warmup steps.
            reward_normalizer: Optional normalizer over intrinsic rewards.
            log_level: Logger level name.
            device: Torch device, or ``None`` for the framework default.
        """
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
        """Build target/predictor ModuleDicts and materialize LazyLinear."""
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
        """Return ``(target_embedding, predictor_embedding)`` for a state batch.

        Args:
            x: Observation batch.

        Returns:
            target_embedding (torch.Tensor): Frozen target output.
            predictor_embedding (torch.Tensor): Predictor output.
        """
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
        """Compute RND learn reward from predictor/target mismatch on next states.

        Args:
            states: Batch of states (unused; kept for API uniformity).
            next_states: Batch of next states scored by RND.
            actions: Unused; kept for API uniformity.
            **kwargs: Extra keyword arguments ignored by RND.

        Returns:
            Per-sample intrinsic reward of shape ``(batch,)``.
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
        """Update the RND predictor on a batch of next states.

        Args:
            states (torch.Tensor): Batch of states (unused).
            next_states (torch.Tensor): Batch of next states to distill.
            actions (torch.Tensor | None): Unused; kept for API uniformity.

        Returns:
            Scalar mean squared prediction error.
        """
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
        """Return a JSON-serializable RND configuration."""
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
        """Rebuild an RND instance from a saved config and weights.

        Args:
            folder (str | pathlib.Path): Parent save directory.
            config (dict): Parsed ``config.json`` contents.
            env (EnvWrapper | None): Optional live env to reuse.

        Returns:
            Restored ``RND`` with loaded ``state_dict``.
        """
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
    """Episodic novelty signal from NGU (Badia et al., 2020).

    Maintains a per-env episodic memory ``M_i``. At each rollout step it:

    1. Embeds ``s_{t+1}`` with an encoder ``φ`` trained by inverse dynamics
       (controllable features only — the same trick as ICM).
    2. Queries ``M_i`` for the k nearest neighbors of ``φ(s_{t+1})``.
    3. Forms the novelty bonus from kernel-summed inverse distances
       ``α_epi = 1 / sqrt(Σ K(φ(s_{t+1}), f_j) + c)`` with
       ``K(x, y) = ε / (d(x,y)² / d̄² + ε)``.
    4. Appends ``φ(s_{t+1})`` to ``M_i``.

    On episode end, ``M_i`` is cleared for env ``i``. Training jointly updates
    ``φ`` and an inverse-dynamics head ``g(φ(s), φ(s')) → â``.

    Attributes:
        memory_size: Cap on embeddings stored per parallel env.
        k: Number of nearest neighbors used in the kernel sum.
        kernel_epsilon: ``ε`` in the inverse-distance kernel.
        cluster_distance: ``c`` added under the square root.
        max_similarity: Square root of the kernel-sum cap; the bonus is
            zeroed once the summed similarity exceeds its square.
        running_mean_decay: EMA decay for the mean squared distance ``d̄``.
        encoder: Feature encoder ``ModuleDict``.
        inverse_model: Inverse-dynamics ``ModuleDict``.
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
        """Build encoder/inverse nets and per-env episodic memories.

        Args:
            env: Environment wrapper; ``num_envs`` sizes the memory list.
            model_configs: Mapping with ``'encoder'`` and ``'inverse_model'``
                entries (``layer_config`` / ``output_layer``).
            optimizer_params: Optimizer configuration for network params.
            memory_size: Maximum embeddings retained per parallel env.
            k: Neighbor count for the episodic kernel.
            kernel_epsilon: Softening constant ``ε`` in the kernel.
            cluster_distance: Additive constant ``c`` under the square root.
            max_similarity: Square root of the summed-similarity cap; the
                bonus becomes zero once the kernel sum exceeds its square.
            running_mean_decay: EMA decay for mean squared neighbor distance.
            reward_weight: Base scale on the episodic bonus.
            reward_scheduler: Optional schedule applied to ``reward_weight``.
            extrinsic_threshold: Intrinsic-only warmup steps.
            reward_normalizer: Optional normalizer over intrinsic rewards.
            log_level: Logger level name.
            device: Torch device, or ``None`` for the framework default.
        """
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
        """Build encoder and inverse-model ModuleDicts and materialize LazyLinear."""
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
        """Encode observations with the episodic encoder.

        Args:
            x: Observation batch.

        Returns:
            Embedding batch from the encoder.
        """
        return self._forward_submodel(x, self.encoder)

    def _knn_bonus(self, embeddings: T.Tensor, env_indices: T.Tensor) -> T.Tensor:
        """Compute per-row episodic novelty bonuses via k-NN kernel sums.

        For each row ``i``, query ``memory[env_indices[i]]`` for the k nearest
        neighbors of ``embeddings[i]`` and return the corresponding ``α_epi``.

        Args:
            embeddings: Embedding batch of shape ``(batch, embed_dim)``.
            env_indices: Parallel-env indices of shape ``(batch,)``.

        Returns:
            Per-sample bonuses of shape ``(batch,)``.
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
        """Append embeddings to the matching per-env episodic memories.

        Args:
            embeddings: Embedding batch to store (moved to CPU).
            env_indices: Parallel-env indices of shape ``(batch,)``.
        """
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
        """Compute rollout episodic novelty and append embeddings to memory.

        Args:
            states: Batch of states (unused; kept for API uniformity).
            next_states: Batch of next states embedded for the bonus.
            actions: Unused; kept for API uniformity.
            env_indices: Parallel-env indices of shape ``(batch,)``. When
                ``None``, indices ``0..batch-1`` are assumed.

        Returns:
            Per-sample rollout reward of shape ``(batch,)``.
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
        """Update encoder and inverse-dynamics head on a transition batch.

        Args:
            states (torch.Tensor): Batch of states.
            next_states (torch.Tensor): Batch of next states.
            actions (torch.Tensor | None): Batch of actions.

        Returns:
            Scalar inverse-dynamics loss.
        """
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
        """Clear episodic memory for envs that just finished.

        Args:
            env_indices: 1-D tensor of finished parallel-env indices.
        """
        for i in env_indices.flatten().tolist():
            self._memories[int(i)] = T.zeros((0, self.embed_dim), dtype=T.float32)

    def get_config(self) -> dict:
        """Return a JSON-serializable EpisodicNovelty configuration."""
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
        """Rebuild an EpisodicNovelty instance from a saved config and weights.

        Args:
            folder (str | pathlib.Path): Parent save directory.
            config (dict): Parsed ``config.json`` contents.
            env (EnvWrapper | None): Optional live env to reuse.

        Returns:
            Restored ``EpisodicNovelty`` with loaded ``state_dict``.
        """
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
    """Combine rewards as a weighted sum ``Σ_i w_i * r_i``.

    Args:
        rewards: Per-module reward tensors of matching shape.
        weights: Optional per-module weights; defaults to ``1.0`` each.

    Returns:
        Combined reward tensor.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        raise ValueError("Empty rewards list")
    if weights is None:
        weights = [1.0] * len(rewards)
    out = weights[0] * rewards[0]
    for w, r in zip(weights[1:], rewards[1:]):
        out = out + w * r
    return out


def multiplicative_combination(rewards: List[T.Tensor]) -> T.Tensor:
    """Combine rewards as a product ``∏_i r_i``.

    Useful when every signal must fire for the bonus to stay large.

    Args:
        rewards: Per-module reward tensors of matching shape.

    Returns:
        Combined reward tensor.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        raise ValueError("Empty rewards list")
    out = rewards[0]
    for r in rewards[1:]:
        out = out * r
    return out


def max_combination(rewards: List[T.Tensor]) -> T.Tensor:
    """Combine rewards by taking the per-step maximum across modules.

    Args:
        rewards: Per-module reward tensors of matching shape.

    Returns:
        Combined reward tensor.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        raise ValueError("Empty rewards list")
    stacked = T.stack(rewards, dim=0)
    return stacked.max(dim=0).values


def ngu_combination(rewards: List[T.Tensor], L: float = 5.0) -> T.Tensor:
    """Combine episodic and lifelong rewards as in NGU.

    Computes ``r = α_epi * clip(α_lifelong, 1, L)``. Expects ``rewards`` in
    order ``[episodic, lifelong]``.

    Args:
        rewards: Exactly two tensors: episodic then lifelong novelty.
        L: Upper clamp for the lifelong factor.

    Returns:
        Combined NGU-style reward tensor.

    Raises:
        ValueError: If ``rewards`` does not contain exactly two tensors.
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
    """Combine several intrinsic-motivation modules under one interface.

    The composite owns no networks and no optimizer — every child manages its
    own training. ``train`` calls each child's ``train`` and returns the sum
    of detached losses (for logging). Per-step rewards are merged with a named
    combination rule (``additive``, ``multiplicative``, ``max``, or ``ngu``).

    Attributes:
        combination_rule: String key into the combination-rule registry.
        combination_kwargs: Extra kwargs for the rule (e.g. ``weights``).
        components: Child ``IntrinsicMotivation`` modules.
        is_online: ``True`` when any child component is online.
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
        """Wrap child modules and select a reward combination rule.

        Args:
            env: Environment wrapper shared with the children.
            components: Child intrinsic-motivation modules to combine.
            combination_rule: One of ``'additive'``, ``'multiplicative'``,
                ``'max'``, or ``'ngu'``.
            combination_kwargs: Extra kwargs for the rule (e.g. ``weights``
                for additive combination).
            reward_weight: Outer scale applied after combining children.
            reward_scheduler: Optional schedule applied to ``reward_weight``.
            extrinsic_threshold: Intrinsic-only warmup steps.
            log_level: Logger level name.
            device: Torch device, or ``None`` for the framework default.
        """
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
            self.is_online = any(c.is_online for c in components)

            # Register components so .to(device), .state_dict() etc. propagate
            for i, c in enumerate(components):
                self.add_module(f"component_{i}", c)

            self.to(self.device)
        except Exception as e:
            self.logger.error(f"Error in CompositeIntrinsicMotivation init: {e}",
                              exc_info=True)
            raise

    def _split_components(self) -> tuple[list[tuple[int, IntrinsicMotivation]], list[tuple[int, IntrinsicMotivation]]]:
        """Split children into online (rollout) and parametric (learn) groups.

        Returns:
            Pair of lists of ``(index, component)``: online first, then
            parametric.
        """
        online, parametric = [], []
        for i, c in enumerate(self.components):
            if c.is_online:
                online.append((i, c))
            else:
                parametric.append((i, c))
        return online, parametric

    def _weights_for(self, components: list[tuple[int, IntrinsicMotivation]]) -> list[float] | None:
        """Select per-component weights from ``combination_kwargs`` by index.

        Args:
            components: List of ``(index, component)`` pairs.

        Returns:
            Weight list aligned with ``components``, or ``None`` when no
            weights were configured.
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
        """Combine rollout rewards from online child modules.

        Args:
            states: Batch of states, shape ``(batch, *state_dim)``.
            next_states: Batch of next states, shape ``(batch, *state_dim)``.
            actions: Batch of actions, shape ``(batch, *action_dim)``, or
                ``None``.
            env_indices: Parallel-env indices of shape ``(batch,)``, or
                ``None``.

        Returns:
            Combined rollout reward of shape ``(batch,)``, or zeros when no
            online children are present.
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
        """Combine learn rewards from parametric children with rollout rewards.

        Args:
            states: Batch of states, shape ``(batch, *state_dim)``.
            next_states: Batch of next states, shape ``(batch, *state_dim)``.
            actions: Batch of actions, shape ``(batch, *action_dim)``, or
                ``None``.
            rollout_rewards: Optional precomputed online/rollout rewards to
                merge with parametric learn rewards.

        Returns:
            Combined learn reward of shape ``(batch,)``, scaled by the
            composite ``reward_weight``.

        Raises:
            RuntimeError: If neither parametric nor rollout rewards are
                available to combine.
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
        """Train every child and return the sum of detached losses.

        Args:
            states (torch.Tensor): Batch of states.
            next_states (torch.Tensor): Batch of next states.
            actions (torch.Tensor | None): Batch of actions.

        Returns:
            Scalar sum of child losses (detached; for logging).
        """
        total = T.tensor(0.0, device=self.device)
        for c in self.components:
            loss = c.train(states, next_states, actions)
            total = total + loss.detach()
        return total

    def on_episode_end(self, env_indices: T.Tensor) -> None:
        """Forward episode-end handling to every child module.

        Args:
            env_indices: 1-D tensor of finished parallel-env indices.
        """
        for c in self.components:
            c.on_episode_end(env_indices)

    def set_normalizers_mode(self, context: Literal['train', 'eval']) -> None:
        """Propagate normalizer mode to every child module.

        Args:
            context: ``'train'`` or ``'eval'``, forwarded to each child.
        """
        for c in self.components:
            c.set_normalizers_mode(context)

    def save(self, folder) -> None:
        """Save each child under its own subdirectory plus a composite config.

        Args:
            folder (str | pathlib.Path): Parent directory for the save tree.
        """
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
        """Return a JSON-serializable composite configuration."""
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
        """Rebuild a composite from saved child directories and config.

        Args:
            folder (str | pathlib.Path): Parent save directory.
            config (dict): Parsed composite ``config.json`` contents.
            env (EnvWrapper | None): Optional live env forwarded to children.

        Returns:
            Restored ``CompositeIntrinsicMotivation``.
        """
        comp_root = Path(folder) / 'intrinsic_motivation'

        sched = (ScheduleWrapper(**config['reward_scheduler'])
                 if config.get('reward_scheduler') else None)

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