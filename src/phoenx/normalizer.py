"""Observation and reward normalizers for training.

Provides running and batch feature normalizers, return-based reward
normalization, a stateless image scaler, and a per-key dict normalizer.
The trainer accumulates samples via ``add``, merges them into running
statistics with ``update``, and applies ``normalize`` when building
model inputs.
"""

import torch as T
import numpy as np
from .torch_utils import get_device
from .logging_config import get_logger


def create_normalizer(config: dict) -> 'BaseNormalizer':
    """Build a normalizer from a ``{type, config}`` mapping.

    Looks up ``config['type']`` in ``NORMALIZER_CLASSES`` and constructs it
    with ``**config['config']``.

    Args:
        config: Mapping with ``type`` (class name string) and ``config``
            (constructor kwargs for that class).

    Returns:
        New normalizer instance of the requested type.

    Raises:
        ValueError: If ``config['type']`` is not in ``NORMALIZER_CLASSES``.
    """
    normalizer_type = config['type']
    if normalizer_type not in NORMALIZER_CLASSES:
        raise ValueError(f"Invalid normalizer type: {normalizer_type}")
    return NORMALIZER_CLASSES[normalizer_type](**config['config'])

class BaseNormalizer:
    """Base class for Welford-style running mean/variance normalizers.

    Local statistics accumulate via ``add``; ``update`` merges them into
    running mean/variance/std and resets the local accumulators. Subclasses
    implement ``normalize``. The ``training`` flag is set by ``train`` /
    ``eval`` and is consulted by subclasses such as ``BatchNorm``.
    """
    def __init__(
        self,
        num_features: int = 1,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        min_std: float = 1e-4,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs
    ):
        """Initialize local and running statistics buffers.

        Args:
            num_features: Feature dimension tracked by the running stats.
            clip_value: Absolute clip bound applied after normalization
                (subclass-dependent).
            epsilon: Small constant added inside the std computation to
                avoid division by zero.
            min_std: Floor for running std used in normalization. Guards
                against near-constant features early in training whose tiny
                std would otherwise amplify later drift (same idea as
                RSL-RL ``EmpiricalNormalization``). Default ``1e-4``
                preserves historical behavior; noisy sim states (e.g. Isaac
                Sim proprioception) typically want ``~1e-2``.
            device: Device for statistics tensors; ``None`` selects the
                framework default via ``get_device``.
            log_level: Logger level name (e.g. ``'INFO'``).
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Extra attributes set on ``self`` via
                ``setattr``.
        """
        self.name = name if name else self.__class__.__name__
        self.logger = get_logger(self.name, level=log_level.upper())
        self.kwargs = kwargs
        self.device = get_device(device)
        self.num_features = num_features
        self.clip_value = T.tensor(clip_value, device=self.device)
        self.epsilon = T.tensor(epsilon, device=self.device)
        self.min_std = float(min_std)
        # Local statistics
        self.local_cnt = T.zeros(1, dtype=T.int32, device=self.device)
        self.local_mean = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.local_M2 = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        # Running statistics
        self.running_cnt = T.zeros(1, dtype=T.int32, device=self.device)
        self.running_mean = T.zeros(self.num_features, dtype=T.float32, device=self.device)
        self.running_var = T.ones(self.num_features, dtype=T.float32, device=self.device)
        self.running_std = T.ones(self.num_features, dtype=T.float32, device=self.device)

        # Set training bool to True
        self.training = True

        # Set internal attributes
        self.step = 0
        self._diag_freq = None
        self._log_diag = False
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def add(self, new_data: T.Tensor) -> None:
        """Accumulate ``new_data`` into local (not yet running) statistics.

        Uses a Welford-style online update of local mean and ``M2``. Does not
        modify running mean/variance; call ``update`` to merge.

        Args:
            new_data: Batch of samples with leading batch dim and trailing
                feature dim ``num_features``.
        """
        self.step += 1
        batch = new_data.to(self.device)
        n = batch.size(0)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_M2 = batch_var * n

        if self.local_cnt.item() == 0:
            self.local_mean = batch_mean
            self.local_M2 = batch_M2
            self.local_cnt += n
        else:
            total = self.local_cnt.item() + n
            delta = batch_mean - self.local_mean
            self.local_mean += delta * (n / total)
            self.local_M2 += batch_M2 + delta**2 * (self.local_cnt.item() * n / total)
            self.local_cnt += n

        # Log diag values if diag
        if self._diag_freq is not None:
            self._log_diag = (self.step % self._diag_freq == 0)
        else:
            self._log_diag = False
        if self._log_diag:
            self.logger.debug(f"Normalizer add: step={self.step}, data={new_data}, data_shape={new_data.shape}, local_cnt={self.local_cnt}, local_mean={self.local_mean}, local_M2={self.local_M2}, running_cnt={self.running_cnt}")

    def update(self) -> None:
        """Merge local statistics into running mean/variance/std and reset local."""
        if self.local_cnt.item() == 0:
            return

        batch_cnt = self.local_cnt.item()
        batch_mean = self.local_mean
        batch_var = self.local_M2 / batch_cnt

        if self.running_cnt.item() == 0:
            self.running_cnt.add_(batch_cnt)
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.running_std = T.sqrt(self.running_var + self.epsilon**2).clamp(min=self.min_std)
        else:
            total_cnt = self.running_cnt + batch_cnt
            delta = batch_mean - self.running_mean

            self.running_mean.add_(delta * (batch_cnt / total_cnt))

            m_a = self.running_var * self.running_cnt
            m_b = batch_var * batch_cnt
            m2 = m_a + m_b + delta**2 * (self.running_cnt * batch_cnt / total_cnt)
            self.running_var.copy_(m2 / total_cnt)

            self.running_cnt.add_(batch_cnt)
            self.running_std = T.sqrt(self.running_var + self.epsilon**2).clamp(min=self.min_std)
        
        if self._log_diag:
            self.logger.debug(f"Normalizer update: step={self.step}, running_cnt={self.running_cnt}, running_mean={self.running_mean}, running_var={self.running_var}, running_std={self.running_std}")

        # Reset local statistics
        self.local_cnt.zero_()
        self.local_mean.zero_()
        self.local_M2.zero_()

    def denormalize(self, v: T.Tensor) -> T.Tensor:
        """Invert running-mean/std normalization.

        Args:
            v: Normalized tensor to map back to the original scale.

        Returns:
            ``v * running_std + running_mean`` on ``self.device``.
        """
        if v.device != self.device:
            v = v.to(self.device)
        return (v * self.running_std) + self.running_mean

    def train(self):
        """Set ``training`` to ``True`` and return ``self``."""
        self.training = True
        return self

    def eval(self):
        """Set ``training`` to ``False`` and return ``self``."""
        self.training = False
        return self

    def get_config(self) -> dict:
        """Return a serializable ``{type, config}`` mapping for this normalizer.

        Returns:
            Mapping with ``type`` set to the class name and ``config`` holding
                constructor fields (``num_features``, ``epsilon``,
                ``clip_value``, ``min_std``, ``device``, ``name``).
        """
        return {
            'type': self.__class__.__name__,
            'config': {
                'num_features':self.num_features,
                'epsilon':self.epsilon.item(),
                'clip_value':self.clip_value.item(),
                'min_std':self.min_std,
                'device':self.device.type,
                'name':self.name,
            },
        }

    def save(self, file_path: str) -> None:
        """Save step counters and local/running statistics to ``file_path``.

        Args:
            file_path: Destination path for ``torch.save``.
        """
        T.save({
            'step': self.step,
            'local_mean': self.local_mean.cpu().detach().numpy(),
            'local_M2': self.local_M2.cpu().detach().numpy(),
            'local_cnt': self.local_cnt.cpu().detach().numpy(),
            'running_cnt': self.running_cnt.cpu().detach().numpy(),
            # 'running_sum': self.running_sum.cpu().detach().numpy(),
            # 'running_sum_sq': self.running_sum_sq.cpu().detach().numpy(),
            'running_mean': self.running_mean.cpu().detach().numpy(),
            'running_var': self.running_var.cpu().detach().numpy(),
            'running_std': self.running_std.cpu().detach().numpy(),
        }, file_path)

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'BaseNormalizer':
        """Dispatch load to the concrete class named by ``config['type']``.

        Accepts the nested ``get_config`` / save format
        (``{'type': ..., 'config': {...}}``).

        Args:
            config: constructor kwargs.
            state_path: Path to a ``torch.save`` statistics file.

        Returns:
            Concrete normalizer instance with statistics restored from
                ``state_path``.

        Raises:
            ValueError: If ``config['type']`` is not in ``NORMALIZER_CLASSES``.
        """
        norm_type = config['type']
        if norm_type not in NORMALIZER_CLASSES:
            raise ValueError(f"Invalid normalizer type: {norm_type}")
        # Subclass .load() expects the flat inner config; the saved/get_config()
        # format nests it under a "config" key. Tolerate an already-flat dict too.
        inner = config.get('config', config)
        return NORMALIZER_CLASSES[norm_type].load(inner, state_path)

    def save_state(self, path) -> None:
        """Persist running/local statistics to ``path`` (creates parent dirs).

        Args:
            path (str | Path): Destination file path.
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(path))

    def load_state(self, path, load_weights: bool = True) -> None:
        """Restore running/local statistics in-place from ``save_state``.

        The architecture is unchanged; only the accumulated statistics are
        overwritten. ``load_weights`` is accepted for interface symmetry with
        the other components and is ignored (a normalizer has no weights).

        Args:
            path (str | Path): Path previously written by ``save_state``.
            load_weights: Ignored; present for API symmetry with agents.
        """
        state = T.load(str(path), map_location='cpu', weights_only=False)
        self.step = state['step']
        self.local_mean = T.as_tensor(state['local_mean'], device=self.device)
        self.local_M2 = T.as_tensor(state['local_M2'], device=self.device)
        self.local_cnt = T.as_tensor(state['local_cnt'], device=self.device)
        self.running_cnt = T.as_tensor(state['running_cnt'], device=self.device)
        self.running_mean = T.as_tensor(state['running_mean'], device=self.device)
        self.running_var = T.as_tensor(state['running_var'], device=self.device)
        self.running_std = T.as_tensor(state['running_std'], device=self.device)

class RunningNorm(BaseNormalizer):
    """Normalize features with running mean and standard deviation.

    ``normalize`` always uses running statistics (independent of the
    ``training`` flag). Call ``add`` then ``update`` to refresh those stats.
    """
    def __init__(
        self,
        num_features: int,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs
    ):
        """Construct a running-mean/std feature normalizer.

        Args:
            num_features: Feature dimension to track.
            clip_value: Absolute clip bound after ``(x - mean) / std``.
            epsilon: Stability constant for the std computation.
            device: Device for statistics tensors.
            log_level: Logger level name.
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Forwarded to ``BaseNormalizer``.
        """
        super().__init__(num_features=num_features, clip_value=clip_value,
                         epsilon=epsilon, device=device, log_level=log_level,
                         name=name, **kwargs)

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """Z-score ``v`` with running mean/std and clip to ``clip_value``.

        Args:
            v: Input tensor with trailing feature dim ``num_features``.

        Returns:
            Clipped normalized tensor as float on ``self.device``.
        """
        if v.device != self.device:
            v = v.to(self.device)

        # if self.training and self.step <= self.warmup_steps:
        #     return v

        norms = T.clamp((v - self.running_mean) / self.running_std,
                       -self.clip_value, self.clip_value).float()
        # Log diag values if diag
        if self._log_diag:
            self.logger.debug(f"RunningNorm normalize: step={self.step}, data={v}, data_shape={v.shape}, running_mean={self.running_mean}, running_std={self.running_std}, norms={norms}")
        return norms

    def get_config(self) -> dict:
        """Return config with ``type`` set to ``RunningNorm``.

        Returns:
            Mapping from ``BaseNormalizer.get_config`` with updated ``type``.
        """
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'RunningNorm':
        """Reconstruct a ``RunningNorm`` and load statistics from disk.

        Args:
            config: Flat constructor kwargs (inner ``config`` from
                ``get_config``).
            state_path: Path to a ``torch.save`` statistics file.

        Returns:
            Instance with local and running statistics restored.
        """
        device = get_device(config['device'])
        state = T.load(state_path, map_location='cpu', weights_only=False)
        normalizer = RunningNorm(
            num_features=config['num_features'],
            clip_value=config['clip_value'],
            epsilon=config['epsilon'],
            device=config['device']
        )
        normalizer.step = state['step']
        normalizer.local_mean = T.tensor(state['local_mean'], device=device)
        normalizer.local_M2 = T.tensor(state['local_M2'], device=device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=device)
        # normalizer.running_sum = T.tensor(state['running_sum'], device=device)
        # normalizer.running_sum_sq = T.tensor(state['running_sum_sq'], device=device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=device)
        normalizer.running_var = T.tensor(state['running_var'], device=device)
        normalizer.running_std = T.tensor(state['running_std'], device=device)

        return normalizer

class BatchNorm(BaseNormalizer):
    """Normalize with batch statistics in train mode, running stats in eval.

    While ``training`` is ``True``, ``normalize`` uses the current batch
    mean/variance (and still clips). In eval mode it uses running mean/std
    without clipping. Running stats are still maintained via ``add`` /
    ``update`` for the eval path.
    """
    def __init__(
        self,
        num_features: int,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs
    ):
        """Construct a batch/running hybrid feature normalizer.

        Args:
            num_features: Feature dimension to track.
            clip_value: Absolute clip bound in train-mode normalization.
            epsilon: Stability constant for the train-mode std computation.
            device: Device for statistics tensors.
            log_level: Logger level name.
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Forwarded to ``BaseNormalizer``.
        """
        super().__init__(num_features=num_features, clip_value=clip_value,
                         epsilon=epsilon, device=device, log_level=log_level,
                         name=name, **kwargs)

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """Normalize ``v`` with batch stats (train) or running stats (eval).

        Args:
            v: Input tensor with trailing feature dim ``num_features``.

        Returns:
            Normalized tensor as float on ``self.device``. Train mode clips
                to ``clip_value``; eval mode does not.
        """
        if v.device != self.device:
            v = v.to(self.device)

        if self.training:
            mean = v.mean(dim=0, keepdim=True)
            var = v.var(dim=0, unbiased=False, keepdim=True)
            std = T.sqrt(var + self.epsilon**2).clamp(min=1e-4)
            norms = T.clamp((v - mean) / std, -self.clip_value, self.clip_value).float()
        else:
            norms = (v - self.running_mean) / self.running_std

        if self._log_diag:
            self.logger.debug(f"BatchNorm normalize: step={self.step}, data={v}, data_shape={v.shape}, running_mean={self.running_mean}, running_std={self.running_std}, norms={norms}")

        return norms

    def get_config(self) -> dict:
        """Return config with ``type`` set to ``BatchNorm``.

        Returns:
            Mapping from ``BaseNormalizer.get_config`` with updated ``type``.
        """
        config = super().get_config()
        config['type'] = self.__class__.__name__
        return config

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'BatchNorm':
        """Reconstruct a ``BatchNorm`` and load statistics from disk.

        Args:
            config: Flat constructor kwargs (inner ``config`` from
                ``get_config``).
            state_path: Path to a ``torch.save`` statistics file.

        Returns:
            Instance with local and running statistics restored.
        """
        device = get_device(config['device'])
        state = T.load(state_path, map_location='cpu', weights_only=False)
        normalizer = BatchNorm(
            num_features=config['num_features'],
            clip_value=config['clip_value'],
            epsilon=config['epsilon'],
            device=config['device']
        )
        normalizer.step = state['step']
        normalizer.local_mean = T.tensor(state['local_mean'], device=device)
        normalizer.local_M2 = T.tensor(state['local_M2'], device=device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=device)
        # normalizer.running_sum = T.tensor(state['running_sum'], device=device)
        # normalizer.running_sum_sq = T.tensor(state['running_sum_sq'], device=device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=device)
        normalizer.running_var = T.tensor(state['running_var'], device=device)
        normalizer.running_std = T.tensor(state['running_std'], device=device)

        return normalizer

class RewardNorm(BaseNormalizer):
    """Normalize rewards by the running standard deviation of discounted returns.

    Maintains per-env discounted returns ``R_t = gamma * R_{t-1} + r_t``,
    feeds those returns into the base local/running statistics, and scales
    raw rewards by ``1 / running_std`` (no mean subtraction). Episode ends
    reset the corresponding env's return to zero.
    """
    def __init__(
        self,
        gamma: float = 0.99,
        clip_value: float = 5.0,
        epsilon: float = 1e-6,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs
    ):
        """Construct a return-std reward normalizer.

        Args:
            gamma: Discount factor for the per-env return tracker.
            clip_value: Absolute clip bound after reward scaling.
            epsilon: Stability constant for the std computation.
            device: Device for statistics and return tensors.
            log_level: Logger level name.
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Forwarded to ``BaseNormalizer``.
        """
        super().__init__(
            clip_value = clip_value,
            epsilon = epsilon,
            device = device,
            log_level = log_level,
            name = name,
            **kwargs
        )
        self.gamma = gamma
        # Set internal attrs
        self.num_envs = None
        self.returns = None

    def add(self, rewards: T.Tensor, dones: T.Tensor):
        """Update discounted returns and feed them into local statistics.

        Lazily allocates a per-env return buffer from ``rewards.shape[0]``.
        After the Welford update, returns for done envs are zeroed.

        Args:
            rewards: Per-env rewards for the current step.
            dones: Boolean mask of finished envs (trainer passes
                terminations OR truncations).
        """
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        
        # Set internal attr if not set
        if self.num_envs is None:
            self.num_envs = rewards.shape[0]
        if self.returns is None and self.num_envs is not None:
            self.returns = T.zeros(self.num_envs, device=self.device)
        
        # Update returns
        self.returns = self.returns * self.gamma + rewards.squeeze(-1)
        super().add(self.returns.unsqueeze(-1))

        if self._log_diag:
            self.logger.debug(f"RewardNorm add: step={self.step}, rewards={rewards}, dones={dones}, returns={self.returns}")

        # Reset env return if done
        self.returns[dones] = 0.0

    def normalize(self, rewards: T.Tensor) -> T.Tensor:
        """Scale rewards by running return std and clip.

        Args:
            rewards: Raw reward tensor to scale.

        Returns:
            ``clamp(rewards / running_std, ±clip_value)`` as float.
        """
        if rewards.device != self.device:
            rewards = rewards.to(self.device)

        norms = T.clamp(rewards / self.running_std, -self.clip_value, self.clip_value).float()

        if self._log_diag:
            self.logger.debug(f"RewardNorm normalize: step={self.step}, data={rewards}, data_shape={rewards.shape}, running_mean={self.running_mean}, running_std={self.running_std}, norms={norms}")
        
        return norms

    def get_config(self) -> dict:
        """Return config including ``gamma``.

        Returns:
            Mapping from ``BaseNormalizer.get_config`` with ``type``
                ``RewardNorm`` and ``gamma`` added under ``config``.
        """
        config = super().get_config()
        config['type'] = self.__class__.__name__
        config['config'].update({
            'gamma': self.gamma,
        })
        return config

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'RewardNorm':
        """Reconstruct a ``RewardNorm`` and load statistics from disk.

        Args:
            config: Flat constructor kwargs including ``gamma``.
            state_path: Path to a ``torch.save`` statistics file.

        Returns:
            Instance with local and running statistics restored.
        """
        device = get_device(config['device'])
        state = T.load(state_path, map_location='cpu', weights_only=False)
        normalizer = RewardNorm(
            gamma=config['gamma'],
            clip_value=config['clip_value'],
            epsilon=config['epsilon'],
            device=config['device']
        )
        normalizer.step = state['step']
        normalizer.local_mean = T.tensor(state['local_mean'], device=device)
        normalizer.local_M2 = T.tensor(state['local_M2'], device=device)
        normalizer.local_cnt = T.tensor(state['local_cnt'], device=device)
        normalizer.running_cnt = T.tensor(state['running_cnt'], device=device)
        normalizer.running_mean = T.tensor(state['running_mean'], device=device)
        normalizer.running_var = T.tensor(state['running_var'], device=device)
        normalizer.running_std = T.tensor(state['running_std'], device=device)

        return normalizer
        

class ImageScale(BaseNormalizer):
    """Stateless image scaler: casts to float and divides by ``scale``.

    Intended for per-key use inside a ``DictNormalizer`` on image modalities
    stored as 0-255 values. Models auto-scale raw ``uint8`` inputs themselves;
    use this when image data reaches the model as floats in the 0-255 range.
    ``add`` / ``update`` are no-ops.
    """

    def __init__(
        self,
        scale: float = 255.0,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs,
    ):
        """Construct a stateless divisor scaler.

        Args:
            scale: Divisor applied in ``normalize`` (default ``255.0``).
            device: Device tensors are moved to before scaling.
            log_level: Logger level name.
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Forwarded to ``BaseNormalizer``.
        """
        super().__init__(num_features=1, clip_value=float('inf'), device=device,
                         log_level=log_level, name=name, **kwargs)
        self.scale = scale

    def add(self, *args, **kwargs) -> None:
        """No-op; image scaling does not track statistics.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.
        """
        pass  # stateless

    def update(self) -> None:
        """No-op; image scaling does not track statistics."""
        pass  # stateless

    def normalize(self, v: T.Tensor) -> T.Tensor:
        """Cast ``v`` to float on ``self.device`` and divide by ``scale``.

        Args:
            v: Image tensor (typically values in ``[0, scale]``).

        Returns:
            Scaled float tensor.
        """
        return v.to(self.device).float() / self.scale

    def denormalize(self, v: T.Tensor) -> T.Tensor:
        """Multiply by ``scale`` (inverse of ``normalize``).

        Args:
            v: Scaled tensor to map back toward the original range.

        Returns:
            ``v * scale`` on ``self.device``.
        """
        return v.to(self.device) * self.scale

    def get_config(self) -> dict:
        """Return ``{type, config}`` with ``scale``, ``device``, and ``name``.

        Returns:
            Serialisable constructor mapping for this scaler.
        """
        return {
            'type': self.__class__.__name__,
            'config': {
                'scale': self.scale,
                'device': self.device.type,
                'name': self.name,
            },
        }

    @classmethod
    def load(cls, config: dict, state_path: str) -> 'ImageScale':
        """Build from ``config``; ``state_path`` is ignored (stateless).

        Args:
            config: Flat kwargs; reads ``scale`` and ``device``.
            state_path: Unused; accepted for interface symmetry.

        Returns:
            New ``ImageScale`` instance.
        """
        return ImageScale(scale=config.get('scale', 255.0), device=config.get('device'))


class DictNormalizer(BaseNormalizer):
    """Per-key normalizer for dict (multi-modal) observations.

    Routes each observation key to its own child normalizer; keys without an
    entry pass through unchanged.

    Config example:

        state_normalizer:
          type: DictNormalizer
          config:
            per_key:
              vec: {type: RunningNorm, config: {num_features: 7, clip_value: 5.0}}
              rgb: {type: ImageScale, config: {}}
    """

    def __init__(
        self,
        per_key: dict,
        device: str | T.device | None = None,
        log_level: str = 'INFO',
        name: str | None = None,
        **kwargs,
    ):
        """Build child normalizers from a per-key config mapping.

        Each ``per_key`` value is a ``{type, config}`` dict passed to
        ``create_normalizer``. Missing ``config.device`` defaults to
        ``device``.

        Args:
            per_key: Mapping from observation key to normalizer config.
            device: Default device forwarded into child configs.
            log_level: Logger level name.
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Forwarded to ``BaseNormalizer``.
        """
        super().__init__(num_features=1, device=device, log_level=log_level,
                         name=name, **kwargs)
        self.normalizers: dict[str, BaseNormalizer] = {}
        for key, cfg in (per_key or {}).items():
            cfg = dict(cfg)
            cfg.setdefault('config', {})
            cfg['config'].setdefault('device', device)
            self.normalizers[key] = create_normalizer(cfg)

    def add(self, new_data: dict) -> None:
        """Call each child's ``add`` for keys present in ``new_data``.

        Args:
            new_data: Observation dict; only configured keys are forwarded.
        """
        for key, norm in self.normalizers.items():
            if key in new_data:
                norm.add(new_data[key])

    def update(self) -> None:
        """Call ``update`` on every child normalizer."""
        for norm in self.normalizers.values():
            norm.update()

    def normalize(self, data: dict) -> dict:
        """Normalize each key that has a child; pass other keys through.

        Args:
            data: Observation dict to normalize.

        Returns:
            New dict with the same keys; configured keys replaced by their
                child's ``normalize`` output.
        """
        return {
            key: (self.normalizers[key].normalize(value) if key in self.normalizers else value)
            for key, value in data.items()
        }

    def train(self):
        """Set this normalizer and every child to train mode.

        Returns:
            normalizer (DictNormalizer): This instance, for chaining.
        """
        super().train()
        for norm in self.normalizers.values():
            norm.train()
        return self

    def eval(self):
        """Set this normalizer and every child to eval mode.

        Returns:
            normalizer (DictNormalizer): This instance, for chaining.
        """
        super().eval()
        for norm in self.normalizers.values():
            norm.eval()
        return self

    def get_config(self) -> dict:
        """Return ``{type, config}`` including each child's ``get_config``.

        Returns:
            Serialisable mapping with ``per_key``, ``device``, and ``name``.
        """
        return {
            'type': self.__class__.__name__,
            'config': {
                'per_key': {key: norm.get_config() for key, norm in self.normalizers.items()},
                'device': self.device.type,
                'name': self.name,
            },
        }

    def save_state(self, path) -> None:
        """Save every child normalizer's statistics into one ``.pt`` file.

        Args:
            path (str | Path): Destination path for the combined state dict.
        """
        from pathlib import Path
        import tempfile, os
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for key, norm in self.normalizers.items():
            # Reuse each child's own save format via a temp file.
            with tempfile.TemporaryDirectory() as tmp:
                child_path = os.path.join(tmp, 'child.pt')
                norm.save(child_path)
                state[key] = T.load(child_path, map_location='cpu', weights_only=False)
        T.save(state, path)

    def load_state(self, path, load_weights: bool = True) -> None:
        """Restore each child's statistics from a combined ``.pt`` file.

        Unknown keys in the file are skipped. ``load_weights`` is accepted for
        API symmetry and ignored by children that have no weights.

        Args:
            path (str | Path): Path previously written by ``save_state``.
            load_weights: Forwarded unused; present for interface symmetry.
        """
        state = T.load(str(path), map_location='cpu', weights_only=False)
        import tempfile, os
        for key, child_state in state.items():
            norm = self.normalizers.get(key)
            if norm is None:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                child_path = os.path.join(tmp, 'child.pt')
                T.save(child_state, child_path)
                norm.load_state(child_path)


NORMALIZER_CLASSES = {
    "RunningNorm": RunningNorm,
    "BatchNorm": BatchNorm,
    "RewardNorm": RewardNorm,
    "ImageScale": ImageScale,
    "DictNormalizer": DictNormalizer,
}
