"""Neural network building blocks for RL policies and critics.

Defines the layer registry and helpers used to assemble networks from YAML
(``build_layer``, weight init, ``build_optimizer``), the legacy ``Model``
hierarchy (policy / value / actor / critic), and ``ModularModel`` with its
roots → trunk → branches layout and typed ``Head`` subclasses.

Custom attention and recurrent layers (``PositionalEncoding``,
``SelfAttention``, ``TransformerEncoderBlock``, ``LazyRecurrent``) register
under the same ``type`` strings configs use. Temporal layers (LSTM / GRU /
causal transformer) belong in the trunk only.
"""

# imports
from abc import abstractmethod
import math
from typing import Optional, List, Dict, Iterator, Callable, Any
from pathlib import Path

import gymnasium as gym
import torch as T
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import torch.nn.functional as F
from torch.distributions import (
    Distribution, TransformedDistribution,
    Categorical, Beta, Normal, Kumaraswamy,
)


from .distributions import SquashedNormal, ScaledBeta, ScaledKumaraswamy, BoundedIndependent
from .torch_utils import get_device, VarianceScaling_
from .logging_config import get_logger
from .env_wrapper import EnvWrapper, GymnasiumWrapper, IsaacSimWrapper
from .schedulers import ScheduleWrapper


# =============================================================================
# Custom layer modules (used by the layer registry)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Adds positional information to a ``(B, T, d_model)`` token sequence."""

    def __init__(self, d_model: int, max_len: int = 512, learned: bool = False):
        """Create a fixed sinusoidal or learned positional embedding.

        Args:
            d_model: Embedding dimension of the tokens.
            max_len: Maximum sequence length supported.
            learned: If True use a learned positional embedding, else sinusoidal.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.learned = learned
        if learned:
            self.pe = nn.Parameter(T.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, std=0.02)
        else:
            position = T.arange(max_len).unsqueeze(1).float()
            div_term = T.exp(T.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = T.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = T.sin(position * div_term)
            pe[0, :, 1::2] = T.cos(position * div_term[: (d_model + 1) // 2])
            self.register_buffer("pe", pe)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Add positional encodings to a batch of token embeddings.

        Args:
            x: Input of shape ``(B, T, d_model)``.

        Returns:
            Tensor of the same shape with positions added.

        Raises:
            ValueError: If ``x`` is not rank-3 or ``T`` exceeds ``max_len``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"positional_encoding expects rank-3 input (B, T, d_model); got shape {tuple(x.shape)}"
            )
        if x.shape[1] > self.max_len:
            raise ValueError(
                f"positional_encoding max_len={self.max_len} but sequence length is {x.shape[1]}"
            )
        return x + self.pe[:, : x.shape[1]]


class SelfAttention(nn.Module):
    """Self-attention wrapper around ``nn.MultiheadAttention`` (batch_first).

    Operates on ``(B, tokens, embed_dim)`` inputs (intra-step attention over
    feature tokens). For temporal (causal) attention use ``transformer_encoder``
    with ``causal: true`` in the trunk.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Wrap a batch-first multi-head attention module.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability on attention weights.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Apply self-attention over the token axis.

        Args:
            x: Input of shape ``(B, tokens, embed_dim)``.

        Returns:
            Attended tensor of the same shape.

        Raises:
            ValueError: If ``x`` is not rank-3.
        """
        if x.dim() != 3:
            raise ValueError(
                f"mha expects rank-3 input (B, tokens, embed_dim); got shape {tuple(x.shape)}"
            )
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class TransformerEncoderBlock(nn.Module):
    """``nn.TransformerEncoder`` wrapper with optional causal and segment masking.

    - ``causal=False``: intra-step attention over feature tokens; allowed in
      any module (roots/trunk/branches).
    - ``causal=True``: temporal attention over the time axis; only allowed in
      the trunk (enforced by ``ModularModel``). When a ``start_mask``
      (``(B, T)`` episode-start flags) is provided, attention is additionally
      blocked across episode boundaries.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        causal: bool = False,
        norm_first: bool = False,
    ):
        """Build a stack of Transformer encoder layers.

        Args:
            d_model: Token / time-step embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of stacked encoder layers.
            dim_feedforward: Hidden size of the feed-forward block.
            dropout: Dropout applied inside each encoder layer.
            activation: Activation name passed to ``nn.TransformerEncoderLayer``.
            causal: If True, apply a causal (and optional episode-segment) mask.
            norm_first: If True, use pre-norm encoder layers.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.causal = causal
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _build_mask(self, x: T.Tensor, start_mask: T.Tensor | None) -> T.Tensor | None:
        """Build the (bool) attention mask: True = blocked."""
        if not self.causal:
            return None
        B, L = x.shape[0], x.shape[1]
        causal = T.triu(T.ones(L, L, dtype=T.bool, device=x.device), diagonal=1)
        if start_mask is None:
            return causal
        # Episode-segment blocking: positions in different episode segments
        # cannot attend to each other. seg id increases at each episode start.
        seg = T.cumsum(start_mask.to(dtype=T.long, device=x.device), dim=1)  # (B, T)
        diff_seg = seg.unsqueeze(2) != seg.unsqueeze(1)  # (B, T, T)
        blocked = causal.unsqueeze(0) | diff_seg  # (B, T, T)
        # Expand to (B * nhead, T, T) as required for 3D attention masks.
        blocked = blocked.repeat_interleave(self.nhead, dim=0)
        return blocked

    def forward(self, x: T.Tensor, start_mask: T.Tensor | None = None) -> T.Tensor:
        """Encode a rank-3 sequence, optionally with causal / segment masks.

        Args:
            x: Input of shape ``(B, T, d_model)``.
            start_mask: Optional ``(B, T)`` bool flags marking episode starts;
                used only when ``causal`` is True to block cross-episode attention.

        Returns:
            Encoded tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``x`` is not rank-3.
        """
        if x.dim() != 3:
            raise ValueError(
                f"transformer_encoder expects rank-3 input (B, T, d_model); got shape {tuple(x.shape)}"
            )
        mask = self._build_mask(x, start_mask)
        return self.encoder(x, mask=mask)


class LazyRecurrent(nn.Module):
    """Lazily-built LSTM/GRU (input size inferred at first forward).

    Mirrors the codebase's lazy-layer convention: the inner ``nn.LSTM``/
    ``nn.GRU`` is materialized during the model's dry run, before weight
    initialization and optimizer construction.
    """

    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_size: int | None = None,
    ):
        """Configure an LSTM or GRU that materializes on first use.

        Args:
            mode: Either ``'lstm'`` or ``'gru'``.
            hidden_size: Hidden (and cell) state size.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout between stacked layers (ignored when
                ``num_layers`` is 1).
            input_size: If given, materialize immediately; otherwise defer
                until the first ``forward``.
        """
        super().__init__()
        if mode not in ("lstm", "gru"):
            raise ValueError(f"LazyRecurrent mode must be 'lstm' or 'gru', got {mode!r}")
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn: nn.Module | None = None
        if input_size is not None:
            self._materialize(input_size)

    def _materialize(self, input_size: int) -> None:
        cls = nn.LSTM if self.mode == "lstm" else nn.GRU
        self.rnn = cls(
            input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, x: T.Tensor, hx=None):
        """Run the recurrent module, materializing it on first call if needed.

        Args:
            x: Input of shape ``(B, T, features)``.
            hx (torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None): Optional
                initial hidden state. For LSTM a ``(h, c)`` pair; for GRU a
                single ``h``. Each tensor has shape
                ``(num_layers, B, hidden_size)``.

        Returns:
            output (torch.Tensor): RNN outputs of shape ``(B, T, hidden_size)``.
            hx (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): Updated hidden
                state in the same layout as the input ``hx``.

        Raises:
            ValueError: If ``x`` is not rank-3.
        """
        if x.dim() != 3:
            raise ValueError(
                f"{self.mode} expects rank-3 input (B, T, features); got shape {tuple(x.shape)}"
            )
        if self.rnn is None:
            self._materialize(x.shape[-1])
            self.rnn.to(x.device)
        return self.rnn(x, hx)

    def init_hidden(self, batch_size: int, device: T.device | str):
        """Allocate a zeroed hidden state for a fresh episode batch.

        Args:
            batch_size: Leading batch dimension ``B``.
            device: Device the zeros are allocated on.

        Returns:
            hidden (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): Zeroed
                state — ``(h, c)`` for LSTM or ``h`` for GRU, each shaped
                ``(num_layers, batch_size, hidden_size)``.
        """
        h = T.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.mode == "lstm":
            return (h, T.zeros_like(h))
        return h

    def mask_hidden(self, hidden, keep_mask: T.Tensor):
        """Zero the hidden state where ``keep_mask`` is False (episode starts).

        Args:
            hidden (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): ``(h, c)``
                for LSTM or ``h`` for GRU; each of shape ``(layers, B, H)``.
            keep_mask: Bool tensor of shape ``(B,)`` — True keeps the state,
                False resets it.
        """
        keep = keep_mask.to(dtype=T.float32).view(1, -1, 1)
        if self.mode == "lstm":
            h, c = hidden
            return (h * keep.to(h.device), c * keep.to(c.device))
        return hidden * keep.to(hidden.device)


# =============================================================================
# Layer registry
# =============================================================================
# Each entry maps a config ``type`` string to a builder ``params -> nn.Module``.
# Builders must not mutate ``params``.

def _build_rmsnorm(params: dict) -> nn.Module:
    if not hasattr(nn, "RMSNorm"):
        raise ValueError("rmsnorm requires torch >= 2.4 (nn.RMSNorm not available)")
    return nn.RMSNorm(**params)


LAYER_REGISTRY: Dict[str, Callable[[dict], nn.Module]] = {
    # Linear
    'dense': lambda p: nn.LazyLinear(p["units"]),
    'linear': lambda p: nn.Linear(p["in_features"], p["out_features"], bias=p.get("bias", True)),
    # Convolutional
    'conv1d': lambda p: nn.LazyConv1d(
        out_channels=p.get('out_channels', 64), kernel_size=p.get('kernel_size', 3),
        stride=p.get('stride', 1), padding=p.get('padding', 0), bias=p.get('bias', True)),
    'conv2d': lambda p: nn.LazyConv2d(
        out_channels=p.get('out_channels', 64), kernel_size=p.get('kernel_size', 3),
        stride=p.get('stride', 1), padding=p.get('padding', 0), bias=p.get('bias', True)),
    'conv3d': lambda p: nn.LazyConv3d(
        out_channels=p.get('out_channels', 64), kernel_size=p.get('kernel_size', 3),
        stride=p.get('stride', 1), padding=p.get('padding', 0), bias=p.get('bias', True)),
    'convtranspose2d': lambda p: nn.LazyConvTranspose2d(
        out_channels=p.get('out_channels', 64), kernel_size=p.get('kernel_size', 3),
        stride=p.get('stride', 1), padding=p.get('padding', 0), bias=p.get('bias', True)),
    # Pooling ('pool' kept for backwards compatibility with existing configs)
    'pool': lambda p: nn.MaxPool2d(**p),
    'maxpool1d': lambda p: nn.MaxPool1d(**p),
    'maxpool2d': lambda p: nn.MaxPool2d(**p),
    'avgpool1d': lambda p: nn.AvgPool1d(**p),
    'avgpool2d': lambda p: nn.AvgPool2d(**p),
    'adaptiveavgpool2d': lambda p: nn.AdaptiveAvgPool2d(p.get('output_size', 1)),
    'adaptivemaxpool2d': lambda p: nn.AdaptiveMaxPool2d(p.get('output_size', 1)),
    # Normalization
    'batchnorm1d': lambda p: nn.LazyBatchNorm1d(**p),
    'batchnorm2d': lambda p: nn.LazyBatchNorm2d(**p),
    'batchnorm3d': lambda p: nn.LazyBatchNorm3d(**p),
    'layernorm': lambda p: nn.LayerNorm(**p),
    'groupnorm': lambda p: nn.GroupNorm(**p),
    'rmsnorm': _build_rmsnorm,
    # Activations
    'relu': lambda p: nn.ReLU(),
    'leakyrelu': lambda p: nn.LeakyReLU(**p),
    'tanh': lambda p: nn.Tanh(),
    'sigmoid': lambda p: nn.Sigmoid(),
    'gelu': lambda p: nn.GELU(**p),
    'silu': lambda p: nn.SiLU(),
    'elu': lambda p: nn.ELU(**p),
    'mish': lambda p: nn.Mish(),
    'softmax': lambda p: nn.Softmax(dim=p.get('dim', -1)),
    'softplus': lambda p: nn.Softplus(**p),
    # Regularization
    'dropout': lambda p: nn.Dropout(**p),
    'dropout2d': lambda p: nn.Dropout2d(**p),
    # Shape
    'flatten': lambda p: nn.Flatten(**p),
    'unflatten': lambda p: nn.Unflatten(p.get('dim', -1), tuple(p['sizes'])),
    # Embedding / attention / sequence
    'embedding': lambda p: nn.Embedding(p['num_embeddings'], p['embedding_dim']),
    'positional_encoding': lambda p: PositionalEncoding(
        d_model=p['d_model'], max_len=p.get('max_len', 512), learned=p.get('learned', False)),
    'mha': lambda p: SelfAttention(
        embed_dim=p['embed_dim'], num_heads=p.get('num_heads', 1), dropout=p.get('dropout', 0.0)),
    'transformer_encoder': lambda p: TransformerEncoderBlock(
        d_model=p['d_model'], nhead=p.get('nhead', 1), num_layers=p.get('num_layers', 1),
        dim_feedforward=p.get('dim_feedforward', 2048), dropout=p.get('dropout', 0.0),
        activation=p.get('activation', 'relu'), causal=p.get('causal', False),
        norm_first=p.get('norm_first', False)),
    'lstm': lambda p: LazyRecurrent(
        'lstm', hidden_size=p['hidden_size'], num_layers=p.get('num_layers', 1),
        dropout=p.get('dropout', 0.0), input_size=p.get('input_size')),
    'gru': lambda p: LazyRecurrent(
        'gru', hidden_size=p['hidden_size'], num_layers=p.get('num_layers', 1),
        dropout=p.get('dropout', 0.0), input_size=p.get('input_size')),
}


#: keys of the per-layer ``params`` dict that configure weight initialization
#: rather than the layer constructor itself.
_INIT_SPEC_KEYS = ('kernel', 'kernel_params', 'kernel params')


def build_layer(layer_type: str, params: dict | None = None) -> nn.Module:
    """Build a layer module from its registry ``type`` and ``params`` dict.

    ``kernel`` / ``kernel_params`` entries (weight-initialization spec) are
    stripped before the remaining params reach the layer constructor.

    Args:
        layer_type: Registry key, e.g. ``'dense'``, ``'lstm'``, ``'mha'``.
        params: Constructor kwargs for that layer type. May also carry
            ``kernel`` / ``kernel_params`` init keys, which are ignored here.

    Returns:
        Instantiated ``nn.Module`` for ``layer_type``.

    Raises:
        ValueError: If ``layer_type`` is unknown or a required param is missing.
    """
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(
            f"Unsupported layer type: {layer_type!r}. Available: {sorted(LAYER_REGISTRY)}"
        )
    ctor_params = {k: v for k, v in (params or {}).items() if k not in _INIT_SPEC_KEYS}
    try:
        return LAYER_REGISTRY[layer_type](ctor_params)
    except KeyError as e:
        raise ValueError(f"Layer type {layer_type!r} is missing required param {e}") from e


# =============================================================================
# Weight initialization helpers
# =============================================================================

def apply_kernel_(tensor: T.Tensor, kernel: str, kernel_params: dict | None = None) -> None:
    """Apply a named initialization scheme in-place to ``tensor``.

    Mirrors (and is shared with) the legacy ``Model._init_weights`` kernel
    dispatch so numerics are identical for existing configs.

    Args:
        tensor: Parameter tensor to initialize in place.
        kernel: Scheme name (``'kaiming_uniform'``, ``'orthogonal'``,
            ``'variance_scaling'``, ``'default'``, …).
        kernel_params: Extra kwargs forwarded to the matching ``nn.init``
            call. Ignored when ``kernel`` is ``'default'``.

    Raises:
        ValueError: If ``kernel`` is not a supported scheme name.
    """
    kernel_params = kernel_params or {}
    if kernel == 'kaiming_uniform':
        nn.init.kaiming_uniform_(tensor, **kernel_params)
    elif kernel == 'kaiming_normal':
        nn.init.kaiming_normal_(tensor, **kernel_params)
    elif kernel == 'xavier_uniform':
        nn.init.xavier_uniform_(tensor, **kernel_params)
    elif kernel == 'xavier_normal':
        nn.init.xavier_normal_(tensor, **kernel_params)
    elif kernel == 'truncated_normal':
        nn.init.trunc_normal_(tensor, **kernel_params)
    elif kernel == 'uniform':
        nn.init.uniform_(tensor, **kernel_params)
    elif kernel == 'normal':
        nn.init.normal_(tensor, **kernel_params)
    elif kernel == 'orthogonal':
        nn.init.orthogonal_(tensor, **kernel_params)
    elif kernel == 'constant':
        nn.init.constant_(tensor, **kernel_params)
    elif kernel == 'ones':
        nn.init.ones_(tensor, **kernel_params)
    elif kernel == 'zeros':
        nn.init.zeros_(tensor, **kernel_params)
    elif kernel == 'variance_scaling':
        VarianceScaling_(tensor, **kernel_params)
    elif kernel == 'default':
        pass  # keep PyTorch's default initialization
    else:
        raise ValueError(f"Unsupported initialization: {kernel}")


def init_module_weights(module: nn.Module, kernel: str = 'default', kernel_params: dict | None = None) -> None:
    """Initialize every weight/bias tensor of ``module`` per the kernel spec.

    Generalizes the legacy single-``weight`` handling to multi-tensor modules
    (LSTM/GRU ``weight_ih_l*``/``weight_hh_l*``, MultiheadAttention in/out
    projections, TransformerEncoderLayer linears):

    - parameters with ``'weight'`` in their name and rank >= 2 get the kernel;
    - rank-1 weights (LayerNorm/BatchNorm scales) are left at their defaults;
    - parameters with ``'bias'`` in their name are zeroed (legacy behavior).

    Args:
        module: Module whose named parameters are initialized in place.
        kernel: Scheme name passed to ``apply_kernel_``. ``'default'`` leaves
            weight tensors at their PyTorch defaults.
        kernel_params: Extra kwargs forwarded to ``apply_kernel_``.
    """
    has_weight = any(
        'weight' in name and p.dim() >= 2 for name, p in module.named_parameters()
    )
    if not has_weight:
        return
    for name, param in module.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if kernel != 'default':
                with T.no_grad():
                    apply_kernel_(param, kernel, kernel_params)
        elif 'bias' in name:
            with T.no_grad():
                nn.init.zeros_(param)


# =============================================================================
# Optimizer builder
# =============================================================================

def build_optimizer(parameters: Iterator[Parameter], optimizer_params: dict) -> optim.Optimizer:
    """Build an optimizer from a ``{'type': ..., 'params': {...}}`` spec.

    Args:
        parameters: Parameter iterator (typically ``module.parameters()``).
        optimizer_params: Spec with required ``type`` (``'adam'``, ``'sgd'``,
            ``'rmsprop'``, or ``'adagrad'``; case-insensitive) and optional
            ``params`` kwargs forwarded to the torch optimizer constructor.

    Returns:
        Instantiated ``torch.optim`` optimizer.

    Raises:
        NotImplementedError: If ``type`` is not one of the supported names.
    """
    original_optimizer_type = optimizer_params['type']
    optimizer_type = str(original_optimizer_type).lower()
    opt_kwargs = optimizer_params.get('params', {})
    if optimizer_type == 'adam':
        return optim.Adam(parameters, **opt_kwargs)
    elif optimizer_type == 'sgd':
        return optim.SGD(parameters, **opt_kwargs)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(parameters, **opt_kwargs)
    elif optimizer_type == 'adagrad':
        return optim.Adagrad(parameters, **opt_kwargs)
    else:
        raise NotImplementedError(f"Unsupported optimizer type: {original_optimizer_type}")


class Model(nn.Module):
    """Legacy single-stack RL model base (superseded by ``ModularModel``).

    Dynamically builds a sequential ``ModuleDict`` from ``layer_config``, then
    lets subclasses attach an output head. Prefer
    [ModularModel][phoenx.models.ModularModel] for new configs (roots → trunk →
    branches).

    Attributes:
        env (EnvWrapper): Environment wrapper providing spaces and keys.
        layer_config (list): Layer specs ``{'type', 'params'}``.
        output_config (dict): Output-layer init / kernel config.
        optimizer_params (dict): Optimizer type and kwargs.
        lr_scheduler (ScheduleWrapper | None): Optional learning-rate schedule.
        device (torch.device): Resolved compute device.
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
        """Build the lazy layer stack and move the module to ``device``.

        Args:
            env: Environment wrapper.
            layer_config: List of dictionaries specifying layers and params.
            output_config: Configuration for output layer initialization.
            optimizer_params: Optimizer configuration; defaults to Adam at
                ``lr=0.001`` when ``None``.
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string or ``None`` (CUDA if available, else CPU).
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
        """Materialize lazy modules with a dummy forward, then init weights.

        Builds synthetic tensors from the env observation (and action, for
        ``ContinuousCritic``) spaces, runs ``forward`` under ``no_grad``, then
        calls ``_init_weights`` on ``module_dict``.

        Args:
            module_dict: ModuleDict whose lazy layers should be materialized.
            layer_config: Parallel list of layer configs used for weight init.
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
        """Build a specific layer based on its type and parameters.

        Args:
            layer_type (str): Type of the layer (e.g., 'dense', 'conv2d', etc.).
            params (dict): Parameters for the layer.

        Returns:
            nn.Module: Constructed layer.
        """
        return build_layer(layer_type, params)

    def _init_weights(self, layer_config, layers):
        """Initialize the weights for the model.

        Args:
            layer_config (dict): configuration of layer.
            layers (torch layers): torch.nn.Module.layers.
        """
        # Loop through each layer config and corresponding layer, applying the
        # configured kernel to every weight tensor of that layer (shared with
        # the modular architecture; handles multi-tensor modules like LSTM/MHA).
        for config, (layer_name, layer) in zip(layer_config, layers.items()):
            kernel = config.get('params', {}).get('kernel', 'default')  # Get kernel or 'default'
            kernel_params = config.get('params', {}).get('kernel_params', {}) # Get kernel params or empty dict
            init_module_weights(layer, kernel, kernel_params)

    def _init_optimizer(self, parameters: Iterator[Parameter] | None = None):
        """Initialize the optimizer for the model.

        Args:
            parameters (Iterator[Parameter] | None): Iterator over the parameters to optimize. If None, uses all parameters.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if parameters is None:
            parameters = self.parameters()
        return build_optimizer(parameters, self.optimizer_params)
    
    def _preprocess_state(self, state):
        """Normalize observation tensor rank and channel layout.

        Adds a feature dim to 1D states, a channel dim to 3D grayscale batches,
        and permutes Gymnasium HWC images to CHW when channels are last.

        Args:
            state (torch.Tensor): Raw observation batch or single observation.

        Returns:
            state (torch.Tensor): Preprocessed observation tensor.
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
        """Recursively unwrap a distribution to get the base distribution (Normal, Beta, etc.).

        Args:
            dist (Distribution): The distribution to unwrap.

        Returns:
            Distribution: The base distribution.
        """
        while True:
            if isinstance(dist, BoundedIndependent):
                dist = dist.base_dist
            elif isinstance(dist, (SquashedNormal, ScaledBeta, ScaledKumaraswamy)):
                dist = dist.base_dist
            elif isinstance(dist, TransformedDistribution):
                dist = dist.base_dist
            else:
                break
        return dist

    def get_mean_actions(self, dist: Distribution)->T.Tensor:
        """Get the mean action of the Transformed distribution.

        Args:
            dist (Distribution): The Transformed distribution to get the mean of.

        Returns:
            mean (torch.Tensor): Deterministic mean / mode action.
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
        """Run the model forward pass (implemented by subclasses)."""
        raise NotImplementedError

    def get_config(self):
        """Serialize architecture hyperparameters to a ``{'type', 'config'}`` dict."""
        return {
            "type": self.__class__.__name__,
            "config": {
                "layer_config": self.layer_config,
                "output_config": self.output_config,
                "optimizer_params": self.optimizer_params,
                "lr_scheduler": self.lr_scheduler.get_config() if self.lr_scheduler else None,
                "device": self.device.type,
            },
        }

    @classmethod
    def from_config(cls, config: dict, env: EnvWrapper) -> "Model":
        """Rebuild architecture + fresh weights from an inner config dict.

        The environment is injected live (never rebuilt here) so a single env
        instance can be shared across every model of an agent. Nested schedule
        dicts for ``lr_scheduler`` / ``temperature_schedule`` are rehydrated via
        ``ScheduleWrapper.from_config``.

        Args:
            config: Inner ``config`` mapping from ``get_config()['config']``.
            env: Live environment wrapper to inject.

        Returns:
            A new instance of ``cls`` with freshly initialized weights.
        """
        import inspect

        cfg = dict(config)
        cfg.pop("env", None)  # env is injected, never taken from the config
        for key in ("lr_scheduler", "temperature_schedule"):
            if isinstance(cfg.get(key), dict):
                cfg[key] = ScheduleWrapper.from_config(cfg[key])
        cfg["env"] = env
        params = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in cfg.items() if k in params}
        return cls(**kwargs)

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Resolve clone device and env; subclasses rebuild the concrete model.

        ``copy_weights`` is accepted for a uniform subclass signature but is not
        used here. Isaac Lab envs are reused; other envs are rebuilt from JSON.

        Args:
            copy_weights: Unused on the base; subclasses may copy weights.
            device: Target device; defaults to this model's device.

        Returns:
            device (torch.device): Resolved device for the clone.
            env (EnvWrapper): Env instance to pass into the subclass constructor.
        """
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

    def save_state(self, path: Path | str) -> None:
        """Write weights + optimizer + scheduler progress to a single ``.pt``.

        Args:
            path: Destination file (e.g. ``.../agent/policy.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "lr_scheduler": self.lr_scheduler.get_state() if self.lr_scheduler is not None else None,
        }
        temperature_schedule = getattr(self, "temperature_schedule", None)
        if temperature_schedule is not None:
            state["temperature_schedule"] = temperature_schedule.get_state()
        T.save(state, path)

    def load_state(self, path: Path | str, load_weights: bool = True) -> None:
        """Restore weights + optimizer + scheduler progress from ``save_state``.

        Args:
            path: Checkpoint path written by ``save_state``.
            load_weights: If False, skip ``load_state_dict`` but still restore
                optimizer / schedule state when present.
        """
        state = T.load(Path(path), map_location=self.device, weights_only=False)
        if load_weights and state.get("model") is not None:
            self.load_state_dict(state["model"])
        if state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("lr_scheduler") is not None and self.lr_scheduler is not None:
            self.lr_scheduler.set_state(state["lr_scheduler"])
        temperature_schedule = getattr(self, "temperature_schedule", None)
        if temperature_schedule is not None and state.get("temperature_schedule") is not None:
            temperature_schedule.set_state(state["temperature_schedule"])


class StochasticDiscretePolicy(Model):
    """Legacy categorical policy for discrete action spaces.

    Modular counterpart: [StochasticDiscreteHead][phoenx.models.StochasticDiscreteHead].

    Attributes:
        distribution (str): Action distribution name (``'categorical'``).
        temperature (float): Softmax temperature multiplier.
        temperature_schedule (ScheduleWrapper | None): Optional temperature
            schedule whose factor scales ``temperature``.
        num_actions (int): Number of discrete actions from the env.
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
        """Initialize the discrete policy, output head, and optimizer.

        Args:
            env: Environment wrapper.
            layer_config: Hidden-layer configuration list.
            output_config: Output-layer weight-init config (list of layer dicts).
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            distribution: Action distribution name; only ``'categorical'``.
            temperature: Base softmax temperature.
            temperature_schedule: Optional schedule multiplying ``temperature``.
            device: Device string/device or ``None`` for the framework default.
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
        """Return a categorical distribution over discrete actions.

        Args:
            x (torch.Tensor): Observation batch (preprocessed in place).
            goal (torch.Tensor | None): Optional goal tensor concatenated on the
                feature dim for goal-conditioned envs.

        Returns:
            dist (torch.distributions.Categorical): Temperature-scaled
                categorical over ``num_actions`` logits.

        Raises:
            ValueError: If ``distribution`` is not ``'categorical'``.
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
        """Serialize including distribution and temperature fields."""
        config = super().get_config()
        config["config"].update({
            'distribution': self.distribution,
            'temperature': self.temperature,
            "temperature_schedule": self.temperature_schedule.get_config() if self.temperature_schedule is not None else None,
        })
        return config

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new discrete policy, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (StochasticDiscretePolicy): Independent policy copy.
        """
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

class StochasticContinuousPolicy(Model):
    """Legacy bounded continuous policy (Beta / Kumaraswamy / squashed Normal).

    Modular counterpart:
    [StochasticContinuousHead][phoenx.models.StochasticContinuousHead].

    Attributes:
        distribution (str): ``'beta'``, ``'kumaraswamy'``, or ``'normal'``.
        num_actions (int): Action dimensionality from the env Box space.
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
        """Initialize the continuous policy, dual-parameter head, and optimizer.

        Args:
            env: Environment wrapper.
            layer_config: Hidden-layer configuration list.
            output_config: Output-layer weight-init config (list of layer dicts).
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            distribution: ``'beta'``, ``'kumaraswamy'``, or ``'normal'``.
            device: Device string/device or ``None`` for the framework default.
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
        """Return a bounded continuous action distribution.

        Args:
            x (torch.Tensor): Observation batch.
            goal (torch.Tensor | None): Optional goal tensor concatenated on the
                feature dim for goal-conditioned envs.

        Returns:
            dist (phoenx.distributions.BoundedIndependent): Scaled Beta,
                Kumaraswamy, or squashed Normal over the action dimensions.

        Raises:
            ValueError: If ``distribution`` is not supported.
        """
         # Preprocess state to ensure correct formatting
        x = self._preprocess_state(x)
        x = x.to(self.device)

        if goal is not None:
            goal = goal.to(self.device)
            x = T.cat([x, goal], dim=-1)

        for layer in self.layers.values():
            x = layer(x)

        param_1 = self.output_layer['policy_output_param_1'](x)
        param_2 = self.output_layer['policy_output_param_2'](x)

        if self.distribution in ['beta', 'kumaraswamy']:
            # Clamp params between -12 and 6 to allow max expressiveness within safe bounds of dist
            param_1 = T.clamp(param_1, min=-12, max=6)
            param_2 = T.clamp(param_2, min=-12, max=6)
            # softplus params to ensure >0 and add 1.0 for numerical stability
            alpha = F.softplus(param_1) + 1.0
            beta = F.softplus(param_2) + 1.0
            # Clamp alpha/beta to prevent exploding gradients
            # alpha = T.clamp(alpha, min=1e-3, max=10.0)
            # beta = T.clamp(beta, min=1e-3, max=10.0)

            low = T.tensor(self.act_space.low, device=self.device)
            high = T.tensor(self.act_space.high, device=self.device)

            if self.distribution == 'beta':
                dist = ScaledBeta(Beta(alpha, beta), low=low, high=high)
        
            elif self.distribution == 'kumaraswamy':
                dist = ScaledKumaraswamy(Kumaraswamy(alpha, beta), low=low, high=high)

        elif self.distribution == 'normal':
            mu = param_1
            # sigma = T.clamp(param_2, min=-6, max=2)
            sigma = F.softplus(param_2) + 1e-6

            low = T.tensor(self.act_space.low, device=self.device)
            high = T.tensor(self.act_space.high, device=self.device)

            dist = SquashedNormal(
                Normal(mu, sigma),
                low=low,
                high=high
            )
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

        return BoundedIndependent(dist, reinterpreted_batch_ndims=1)

    def get_config(self):
        """Serialize including the continuous distribution name."""
        config = super().get_config()
        config["config"].update({
            'distribution': self.distribution,
        })
        return config

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new continuous policy, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (StochasticContinuousPolicy): Independent policy copy.
        """
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


class ValueModel(Model):
    """Legacy scalar state-value network V(s).

    Modular counterpart: [ValueHead][phoenx.models.ValueHead].
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
        """Initialize the value network, scalar head, and optimizer.

        Args:
            env: Environment wrapper.
            layer_config: Hidden-layer configuration list.
            output_config: Output-layer weight-init config.
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string/device or ``None`` for the framework default.
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
        """Predict a scalar state value.

        Args:
            x (torch.Tensor): Observation batch.
            goal (torch.Tensor | None): Optional goal tensor concatenated on the
                feature dim for goal-conditioned envs.

        Returns:
            value (torch.Tensor): Predicted state values with shape ``[..., 1]``.
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
        """Serialize architecture hyperparameters (delegates to ``Model``)."""
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new value model, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (ValueModel): Independent value-network copy.
        """
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

class ActorModel(Model):
    """Legacy deterministic continuous actor returning ``(mu, pi)``.

    Modular counterpart:
    [DeterministicActorHead][phoenx.models.DeterministicActorHead].
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
        """Initialize the deterministic actor, tanh head, and optimizer.

        Args:
            env: Environment wrapper with a continuous Box action space.
            layer_config: Hidden-layer configuration list.
            output_config: Output-layer weight-init config.
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string/device or ``None`` for the framework default.
        """
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
        """Return unbounded ``mu`` and bound-scaled ``pi`` actions.

        Args:
            x (torch.Tensor): Observation batch.
            goal (torch.Tensor | None): Optional goal tensor concatenated on the
                feature dim for goal-conditioned envs.

        Returns:
            mu (torch.Tensor): Pre-tanh linear outputs.
            pi (torch.Tensor): Tanh actions scaled into ``[low, high]`` when
                bounds are finite; otherwise raw tanh outputs in ``[-1, 1]``.
        """
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
        """Serialize architecture hyperparameters (delegates to ``Model``)."""
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new actor, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (ActorModel): Independent actor copy.
        """
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

class BaseCritic(Model):
    """Legacy critic base shared by continuous and discrete Q-networks."""

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
        """Initialize the shared critic body (no output head yet).

        Args:
            env: Environment wrapper.
            layer_config: Hidden-layer configuration list for the state trunk.
            output_config: Output-layer weight-init config (used by subclasses).
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string/device or ``None`` for the framework default.
        """
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Predict action values (implemented by subclasses)."""
        raise NotImplementedError

    def get_config(self):
        """Serialize architecture hyperparameters (delegates to ``Model``)."""
        return super().get_config()

    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Resolve clone device/env like ``Model.clone`` (subclasses rebuild).

        Args:
            copy_weights: Unused here; subclasses may copy weights.
            device: Target device; defaults to this model's device.

        Returns:
            device (torch.device): Resolved device for the clone.
            env (EnvWrapper): Env instance for the subclass constructor.
        """
        return super().clone(copy_weights, device)

class ContinuousCritic(BaseCritic):
    """Legacy Q(s, a) critic for continuous actions.

    Pipeline: state stack → concat(action) → merged stack → scalar Q.
    Modular counterpart: [ContinuousQHead][phoenx.models.ContinuousQHead].
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
        """Initialize state, merged, and output stacks plus the optimizer.

        Args:
            env: Environment wrapper with a continuous Box action space.
            layer_config: Hidden layers applied to the (optionally goal-augmented)
                state before the action is concatenated.
            merged_config: Hidden layers applied after ``concat(state, action)``.
            output_config: Output-layer weight-init config for the scalar head.
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string/device or ``None`` for the framework default.
        """
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
        """Predict a scalar Q(s, a) value.

        Args:
            state (torch.Tensor): Observation batch.
            action (torch.Tensor): Action batch concatenated after the state
                stack.
            goal (torch.Tensor | None): Optional goal tensor concatenated onto
                ``state`` before the state stack.

        Returns:
            q (torch.Tensor): Scalar Q-values with shape ``[..., 1]``.
        """
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
        """Serialize including ``merged_config``."""
        config = super().get_config()
        config["config"].update({
            'merged_config': self.merged_config,
        })

        return config
    
    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new continuous critic, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (ContinuousCritic): Independent critic copy.
        """
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

class DiscreteCritic(BaseCritic):
    """Legacy Q(s, ·) critic over discrete actions.

    Modular counterpart: [DiscreteQHead][phoenx.models.DiscreteQHead].
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
        """Initialize the discrete Q-network, action head, and optimizer.

        Args:
            env: Environment wrapper with a Discrete action space.
            layer_config: Hidden-layer configuration list.
            output_config: Output-layer weight-init config.
            optimizer_params: Optimizer type and params (default Adam ``lr=0.001``).
            lr_scheduler: Optional learning-rate scheduler.
            device: Device string/device or ``None`` for the framework default.
        """
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
        """Predict a Q-value vector over discrete actions.

        Args:
            state (torch.Tensor): Observation batch.
            goal (torch.Tensor | None): Optional goal tensor concatenated on the
                feature dim for goal-conditioned envs.

        Returns:
            q (torch.Tensor): Per-action Q-values with shape
                ``[..., n_actions]``.
        """
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
        """Serialize architecture hyperparameters (delegates to ``Model``)."""
        config = super().get_config()
        return config
    
    def clone(self, copy_weights: bool = True, device: Optional[str | T.device] = None):
        """Return a new discrete critic, optionally copying weights.

        Args:
            copy_weights: If True, copy ``state_dict`` into the clone.
            device: Device for the clone; defaults to this model's device.

        Returns:
            cloned_model (DiscreteCritic): Independent critic copy.
        """
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


# =============================================================================
# Modular (roots -> trunk -> branches) architecture
# =============================================================================

#: Reserved input key under which the ``goal`` tensor passed to
#: :meth:`ModularModel.forward` is injected for Dict-routed roots.
GOAL_INPUT_KEY = "goal"


def unwrap_distribution(dist: Distribution) -> Distribution:
    """Recursively unwrap a distribution to its base (Normal, Beta, etc.).

    Peels ``BoundedIndependent``, ``SquashedNormal``, ``ScaledBeta``,
    ``ScaledKumaraswamy``, and ``TransformedDistribution`` wrappers until a
    non-wrapped base distribution remains.

    Args:
        dist: Possibly wrapped distribution.

    Returns:
        The innermost base distribution.
    """
    while True:
        if isinstance(dist, BoundedIndependent):
            dist = dist.base_dist
        elif isinstance(dist, (SquashedNormal, ScaledBeta, ScaledKumaraswamy)):
            dist = dist.base_dist
        elif isinstance(dist, TransformedDistribution):
            dist = dist.base_dist
        else:
            break
    return dist


def mean_actions_from_dist(dist: Distribution, act_space, device) -> T.Tensor:
    """Deterministic mean action of a (possibly wrapped) distribution.

    Mirrors ``Model.get_mean_actions`` exactly: scales Beta/Kumaraswamy means
    to the action bounds, applies tanh-squash decoding for Normal, and returns
    the mode for Categorical.

    Args:
        dist: Policy distribution (may be wrapped).
        act_space (gymnasium.Space): Action space providing ``low``/``high``
            for continuous bases.
        device (torch.device | str): Device for bound tensors.

    Returns:
        Deterministic action tensor.

    Raises:
        ValueError: If the unwrapped base distribution type is unsupported.
    """
    base_dist = unwrap_distribution(dist)
    if isinstance(base_dist, (Normal, Beta, Kumaraswamy)):
        low = T.tensor(act_space.low, dtype=T.float32, device=device)
        high = T.tensor(act_space.high, dtype=T.float32, device=device)
        if isinstance(base_dist, (Beta, Kumaraswamy)):
            return low + (high - low) * base_dist.mean
        mu = base_dist.loc
        scale = (high - low) / 2.0
        loc = (high + low) / 2.0
        return loc + scale * T.tanh(mu)
    elif isinstance(base_dist, Categorical):
        return base_dist.mode
    else:
        raise ValueError(f"Unsupported distribution: {type(base_dist)}")


class SubNetwork(nn.Module):
    """Named layer stack used as a root, trunk, or head body.

    Built from a ``layer_config`` list of ``{'type', 'params'}`` dicts. An
    empty config is an identity. Temporal layers (LSTM / GRU / causal
    transformer) may appear here when the stack is the trunk only —
    ``ModularModel`` rejects them in roots and branch bodies.
    """

    def __init__(
        self,
        layer_config: List[Dict] | None = None,
        input_keys: List[str] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        name: str | None = None,
    ):
        """Build the layer stack and record temporal-layer keys.

        Args:
            layer_config: List of ``{'type', 'params'}`` layer dicts; empty or
                ``None`` yields an identity stack.
            input_keys: For roots only — observation dict keys this
                sub-network consumes (multiple keys are concatenated along the
                feature dim). The reserved key ``'goal'`` routes the goal
                tensor passed to ``ModularModel.forward``.
            optimizer_params: Optional per-module optimizer spec; ``None``
                inherits the model-level default.
            lr_scheduler: Optional per-module ``ScheduleWrapper``.
            name: Display name used in error messages.
        """
        super().__init__()
        self.layer_config = list(layer_config) if layer_config else []
        self.input_keys = list(input_keys) if input_keys else None
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.name = name or self.__class__.__name__

        self.layers = nn.ModuleDict()
        for i, layer_info in enumerate(self.layer_config):
            layer_type = layer_info['type']
            layer_params = layer_info.get('params', {})
            self.layers[f'{layer_type}_{i}'] = build_layer(layer_type, layer_params)

        self._recurrent_keys = [
            k for k, m in self.layers.items() if isinstance(m, LazyRecurrent)
        ]
        self._causal_keys = [
            k for k, m in self.layers.items()
            if isinstance(m, TransformerEncoderBlock) and m.causal
        ]

    #: layer types that imply image-like (channelled) inputs; used to gate the
    #: legacy grayscale/HWC preprocessing heuristics per root.
    _IMAGE_LAYER_TYPES = frozenset({
        'conv1d', 'conv2d', 'conv3d', 'convtranspose2d', 'pool', 'maxpool1d',
        'maxpool2d', 'avgpool1d', 'avgpool2d', 'adaptiveavgpool2d',
        'adaptivemaxpool2d', 'batchnorm2d', 'batchnorm3d', 'dropout2d',
    })

    @property
    def is_recurrent(self) -> bool:
        """True when the stack contains one or more ``LazyRecurrent`` layers."""
        return len(self._recurrent_keys) > 0

    @property
    def is_causal(self) -> bool:
        """True when the stack contains a causal ``TransformerEncoderBlock``."""
        return len(self._causal_keys) > 0

    @property
    def is_temporal(self) -> bool:
        """True when the stack is recurrent or causal (temporal)."""
        return self.is_recurrent or self.is_causal

    @property
    def expects_image(self) -> bool:
        """True when the stack contains convolutional/pooling layers."""
        return any(info['type'] in self._IMAGE_LAYER_TYPES for info in self.layer_config)

    @property
    def expects_tokens(self) -> bool:
        """True when the first layer is an embedding (integer token input)."""
        return bool(self.layer_config) and self.layer_config[0]['type'] == 'embedding'

    def init_hidden(self, batch_size: int, device: T.device | str) -> Dict[str, Any]:
        """Allocate zero hidden states for every recurrent layer.

        Args:
            batch_size: Leading batch size for the states.
            device: Device the zero tensors are allocated on.

        Returns:
            Mapping of layer name to that layer's zero state.
        """
        return {
            k: self.layers[k].init_hidden(batch_size, device) for k in self._recurrent_keys
        }

    def init_weights(self) -> None:
        """Apply each layer's kernel spec (multi-tensor aware)."""
        for config, (layer_name, layer) in zip(self.layer_config, self.layers.items()):
            kernel = config.get('params', {}).get('kernel', 'default')
            kernel_params = config.get('params', {}).get('kernel_params', {})
            init_module_weights(layer, kernel, kernel_params)

    def forward(
        self,
        x: T.Tensor,
        hidden: Dict[str, Any] | None = None,
        start_mask: T.Tensor | None = None,
        mode: str = 'step',
    ) -> tuple[T.Tensor, Dict[str, Any]]:
        """Run the layer stack.

        Args:
            x: ``(B, ...)`` in step mode / ``(B, T, ...)`` in sequence mode.
            hidden: Recurrent states keyed by layer name (zeros when absent).
            start_mask: Episode-start flags — ``(B,)`` bool in step mode,
                ``(B, T)`` bool in sequence mode. Hidden states are reset
                (zeroed) at flagged positions before they are consumed.
            mode: ``'step'`` or ``'sequence'``.

        Returns:
            output (torch.Tensor): Stack output with the same leading layout
                as ``x``.
            new_hidden (dict): Recurrent states keyed by layer name.
        """
        new_hidden: Dict[str, Any] = {}
        for key, layer in self.layers.items():
            if isinstance(layer, LazyRecurrent):
                h0 = hidden.get(key) if hidden else None
                x, h_new = self._forward_recurrent(layer, x, h0, start_mask, mode)
                new_hidden[key] = h_new
            elif isinstance(layer, TransformerEncoderBlock) and layer.causal:
                if mode == 'sequence':
                    x = layer(x, start_mask=start_mask)
                else:
                    # Step mode: a causal layer degenerates to a length-1 window.
                    # Rolling inference context is provided by the agent (Phase 5).
                    x = layer(x.unsqueeze(1)).squeeze(1)
            else:
                x = layer(x)
        return x, new_hidden

    def _forward_recurrent(
        self,
        layer: LazyRecurrent,
        x: T.Tensor,
        h0,
        start_mask: T.Tensor | None,
        mode: str,
    ):
        if mode == 'step':
            batch_size = x.shape[0]
            if h0 is None:
                h0 = layer.init_hidden(batch_size, x.device)
            if start_mask is not None:
                h0 = layer.mask_hidden(h0, ~start_mask.bool())
            out, h_new = layer(x.unsqueeze(1), h0)
            return out.squeeze(1), h_new

        # Sequence mode: x is (B, T, F)
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h0 is None:
            h0 = layer.init_hidden(batch_size, x.device)
        if start_mask is None or not bool(start_mask[:, 1:].any()):
            # Fast path: no mid-sequence episode boundaries.
            if start_mask is not None:
                h0 = layer.mask_hidden(h0, ~start_mask[:, 0].bool())
            return layer(x, h0)
        # Exact path: reset hidden at every episode start inside the window.
        outs = []
        h = h0
        for t in range(seq_len):
            h = layer.mask_hidden(h, ~start_mask[:, t].bool())
            out_t, h = layer(x[:, t:t + 1], h)
            outs.append(out_t)
        return T.cat(outs, dim=1), h


# -----------------------------------------------------------------------------
# Branch heads
# -----------------------------------------------------------------------------

class Head(nn.Module):
    """Base class for branch heads: a ``SubNetwork`` body plus env-shaped outputs.

    ``output_config`` retains its legacy meaning: the initialization spec of
    the auto-created output layer(s) (widths come from the env). Heads receive
    already-encoded features from the composite model, so unlike the legacy
    ``Model`` subclasses they never touch raw observations.
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict] | None = None,
        output_config: List[Dict] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        device: str | T.device | None = None,
        name: str | None = None,
    ):
        """Attach a body stack and build env-shaped output layers.

        Args:
            env: Environment wrapper providing observation and action spaces.
            layer_config: Hidden-layer config for the head body (may be empty).
            output_config: Init specs for the auto-created output layer(s);
                defaults to a single dense layer with default kernel.
            optimizer_params: Optional per-head optimizer spec.
            lr_scheduler: Optional per-head ``ScheduleWrapper``.
            device: Device to place parameters on (``None`` = framework default).
            name: Display name used in error messages.
        """
        super().__init__()
        self.env = env
        self.name = name or self.__class__.__name__
        self.layer_config = list(layer_config) if layer_config else []
        self.output_config = list(output_config) if output_config else [
            {'type': 'dense', 'params': {'kernel': 'default', 'kernel_params': {}}}
        ]
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.device = get_device(device)
        self.logger = get_logger(self.__class__.__name__, level='INFO')

        self.obs_space = (env.single_observation_space
                          if hasattr(env, 'single_observation_space')
                          else env.observation_space)
        self.act_space = (env.single_action_space
                          if hasattr(env, 'single_action_space')
                          else env.action_space)

        self.body = SubNetwork(self.layer_config, name=f'{self.name}.body')
        self.output_layer = nn.ModuleDict()
        self._build_output_layer()
        self.to(self.device)

    def _build_output_layer(self) -> None:
        raise NotImplementedError

    def init_weights(self) -> None:
        """Initialize body and output layers from their configs.

        Body layers use ``layer_config`` kernels; output layers use
        ``output_config`` (legacy zip semantics preserved).
        """
        self.body.init_weights()
        for config, (layer_name, layer) in zip(self.output_config, self.output_layer.items()):
            kernel = config.get('params', {}).get('kernel', 'default')
            kernel_params = config.get('params', {}).get('kernel_params', {})
            init_module_weights(layer, kernel, kernel_params)

    def _body_forward(self, features: T.Tensor) -> T.Tensor:
        y, _ = self.body(features)
        return y

    def forward(self, features: T.Tensor, action: T.Tensor | None = None):
        """Map trunk features (and optional action) to this head's output.

        Args:
            features: Encoded features from roots+trunk.
            action: Action tensor required by action-conditioned heads
                (e.g. ``ContinuousQHead``); ignored by others.

        Raises:
            NotImplementedError: Always; concrete heads override this.
        """
        raise NotImplementedError

    # -- distribution helpers (mirroring the legacy Model API) ----------------
    def _unwrap_distribution(self, dist: Distribution) -> Distribution:
        return unwrap_distribution(dist)

    def get_mean_actions(self, dist: Distribution) -> T.Tensor:
        """Deterministic mean action for ``dist`` under this head's action space.

        Args:
            dist: Policy distribution produced by a stochastic head.

        Returns:
            Deterministic action tensor on this head's device.
        """
        return mean_actions_from_dist(dist, self.act_space, self.device)

    def get_config(self) -> dict:
        """Serialize this head to a ``{'type', 'config'}`` dict."""
        return {
            'type': self.__class__.__name__,
            'config': {
                'layer_config': self.layer_config,
                'output_config': self.output_config,
                'optimizer_params': self.optimizer_params,
                'lr_scheduler': self.lr_scheduler.get_config() if self.lr_scheduler else None,
                'device': self.device.type,
            },
        }

    @classmethod
    def from_config(cls, config: dict, env: EnvWrapper) -> 'Head':
        """Rebuild a head from an inner config dict, injecting ``env``.

        Schedule dicts under ``lr_scheduler`` / ``temperature_schedule`` are
        reconstructed as ``ScheduleWrapper`` instances. Keys unknown to the
        constructor are dropped.

        Args:
            config: Inner config (the ``'config'`` payload of ``get_config``).
            env: Live environment wrapper injected into the constructor.

        Returns:
            A new head instance of this class.
        """
        import inspect
        cfg = dict(config)
        cfg.pop('env', None)
        for key in ('lr_scheduler', 'temperature_schedule'):
            if isinstance(cfg.get(key), dict):
                cfg[key] = ScheduleWrapper.from_config(cfg[key])
        cfg['env'] = env
        params = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in cfg.items() if k in params}
        return cls(**kwargs)


class ValueHead(Head):
    """State-value head ``V(s)`` (modular counterpart of ``ValueModel``)."""

    def _build_output_layer(self) -> None:
        self.output_layer = nn.ModuleDict({'value_dense_output': nn.LazyLinear(1)})

    def forward(self, features: T.Tensor, action: T.Tensor | None = None) -> T.Tensor:
        """Predict a scalar state value from encoded features.

        Args:
            features: Encoded features from roots+trunk.
            action: Unused; accepted for a uniform head call signature.

        Returns:
            Value tensor of shape ``(..., 1)``.
        """
        x = self._body_forward(features)
        return self.output_layer['value_dense_output'](x)


class StochasticDiscreteHead(Head):
    """Categorical policy head (modular counterpart of ``StochasticDiscretePolicy``)."""

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict] | None = None,
        output_config: List[Dict] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        distribution: str = 'categorical',
        temperature: float = 1.0,
        temperature_schedule: ScheduleWrapper | None = None,
        device: str | T.device | None = None,
        name: str | None = None,
    ):
        """Configure logits temperature and build the categorical output layer.

        Args:
            env: Environment wrapper providing a discrete action space.
            layer_config: Hidden-layer config for the head body.
            output_config: Init specs for the logits output layer.
            optimizer_params: Optional per-head optimizer spec.
            lr_scheduler: Optional per-head ``ScheduleWrapper``.
            distribution: Distribution name; only ``'categorical'`` is supported.
            temperature: Base softmax temperature applied to logits.
            temperature_schedule: Optional multiplier schedule on temperature.
            device: Device to place parameters on.
            name: Display name used in error messages.
        """
        self.distribution = distribution
        self.temperature = temperature
        self.temperature_schedule = temperature_schedule
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device, name)

    def _build_output_layer(self) -> None:
        self.num_actions = self.act_space.n
        self.output_layer = nn.ModuleDict({'policy_dense_output': nn.LazyLinear(self.num_actions)})

    def forward(self, features: T.Tensor, action: T.Tensor | None = None) -> Categorical:
        """Return a categorical distribution over discrete actions.

        Args:
            features: Encoded features from roots+trunk.
            action: Unused; accepted for a uniform head call signature.

        Returns:
            ``Categorical`` over ``act_space.n`` logits (temperature-scaled).

        Raises:
            ValueError: If ``distribution`` is not ``'categorical'``.
        """
        x = self._body_forward(features)
        x = self.output_layer['policy_dense_output'](x)
        if self.distribution == 'categorical':
            temperature = self.temperature
            if self.temperature_schedule is not None:
                temperature *= self.temperature_schedule.get_factor()
            return Categorical(logits=x / temperature)
        raise ValueError(f'Distribution {self.distribution} not supported.')

    def get_config(self) -> dict:
        """Serialize including distribution and temperature fields."""
        config = super().get_config()
        config['config'].update({
            'distribution': self.distribution,
            'temperature': self.temperature,
            'temperature_schedule': self.temperature_schedule.get_config()
            if self.temperature_schedule is not None else None,
        })
        return config


class StochasticContinuousHead(Head):
    """Bounded continuous policy head (modular counterpart of ``StochasticContinuousPolicy``).

    Output math matches the legacy policy: Beta / Kumaraswamy / squashed Normal
    wrapped in ``BoundedIndependent``.
    """

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict] | None = None,
        output_config: List[Dict] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        distribution: str = 'beta',
        device: str | T.device | None = None,
        name: str | None = None,
    ):
        """Select the continuous distribution family and build param heads.

        Args:
            env: Environment wrapper providing a Box action space.
            layer_config: Hidden-layer config for the head body.
            output_config: Init specs for the two parameter output layers.
            optimizer_params: Optional per-head optimizer spec.
            lr_scheduler: Optional per-head ``ScheduleWrapper``.
            distribution: ``'beta'``, ``'kumaraswamy'``, or ``'normal'``.
            device: Device to place parameters on.
            name: Display name used in error messages.
        """
        self.distribution = distribution
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device, name)

    def _build_output_layer(self) -> None:
        self.num_actions = self.act_space.shape[-1]
        self.output_layer = nn.ModuleDict({
            'policy_output_param_1': nn.LazyLinear(self.num_actions),
            'policy_output_param_2': nn.LazyLinear(self.num_actions),
        })

    def forward(self, features: T.Tensor, action: T.Tensor | None = None):
        """Return a bounded continuous action distribution.

        Args:
            features: Encoded features from roots+trunk.
            action: Unused; accepted for a uniform head call signature.

        Returns:
            dist (phoenx.distributions.BoundedIndependent): Scaled Beta,
                Kumaraswamy, or squashed Normal over the action dimensions.

        Raises:
            ValueError: If ``distribution`` is not supported.
        """
        x = self._body_forward(features)
        param_1 = self.output_layer['policy_output_param_1'](x)
        param_2 = self.output_layer['policy_output_param_2'](x)

        if self.distribution in ['beta', 'kumaraswamy']:
            # Clamp params between -12 and 6 to allow max expressiveness within safe bounds of dist
            param_1 = T.clamp(param_1, min=-12, max=6)
            param_2 = T.clamp(param_2, min=-12, max=6)
            # softplus params to ensure >0 and add 1.0 for numerical stability
            alpha = F.softplus(param_1) + 1.0
            beta = F.softplus(param_2) + 1.0

            low = T.tensor(self.act_space.low, device=param_1.device)
            high = T.tensor(self.act_space.high, device=param_1.device)

            if self.distribution == 'beta':
                dist = ScaledBeta(Beta(alpha, beta), low=low, high=high)
            else:
                dist = ScaledKumaraswamy(Kumaraswamy(alpha, beta), low=low, high=high)

        elif self.distribution == 'normal':
            mu = param_1
            # Clamp the pre-softplus scale like the beta branch above: bounds
            # sigma to [softplus(-12)+1e-6, softplus(6)+1e-6] ~ [7e-6, 6.0] so
            # extreme features can neither collapse sigma to 0 (inf log-probs)
            # nor overflow softplus.
            param_2 = T.clamp(param_2, min=-12, max=6)
            sigma = F.softplus(param_2) + 1e-6

            low = T.tensor(self.act_space.low, device=param_1.device)
            high = T.tensor(self.act_space.high, device=param_1.device)

            dist = SquashedNormal(Normal(mu, sigma), low=low, high=high)
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

        return BoundedIndependent(dist, reinterpreted_batch_ndims=1)

    def get_config(self) -> dict:
        """Serialize including the continuous distribution name."""
        config = super().get_config()
        config['config'].update({'distribution': self.distribution})
        return config


class DeterministicActorHead(Head):
    """Deterministic actor head ``(mu, pi)`` (modular counterpart of ``ActorModel``)."""

    def _build_output_layer(self) -> None:
        self.act_space_low = T.tensor(self.act_space.low, dtype=T.float32, device=self.device)
        self.act_space_high = T.tensor(self.act_space.high, dtype=T.float32, device=self.device)
        self.num_actions = self.act_space.shape[-1]
        self.output_layer = nn.ModuleDict({
            'actor_mu': nn.LazyLinear(self.num_actions),
            'actor_pi': nn.Tanh(),
        })

    def forward(self, features: T.Tensor, action: T.Tensor | None = None):
        """Return unbounded ``mu`` and bound-scaled ``pi`` actions.

        Args:
            features: Encoded features from roots+trunk.
            action: Unused; accepted for a uniform head call signature.

        Returns:
            mu (torch.Tensor): Pre-tanh linear outputs.
            pi (torch.Tensor): Tanh actions scaled into ``[low, high]`` when
                bounds are finite; otherwise raw tanh outputs in ``[-1, 1]``.
        """
        x = self._body_forward(features)
        mu = self.output_layer['actor_mu'](x)
        pi = self.output_layer['actor_pi'](mu)
        if not T.isinf(self.act_space_high).any() and not T.isinf(self.act_space_low).any():
            low = self.act_space_low.to(pi.device)
            high = self.act_space_high.to(pi.device)
            pi = low + (pi + 1.0) * 0.5 * (high - low)
        return mu, pi


class ContinuousQHead(Head):
    """Q(s, a) head for continuous actions (modular counterpart of ``ContinuousCritic``).

    Pipeline: state body stack → concat(action) → merged stack → scalar Q.
    """

    requires_action = True

    def __init__(
        self,
        env: EnvWrapper,
        layer_config: List[Dict] | None = None,
        merged_config: List[Dict] | None = None,
        output_config: List[Dict] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        device: str | T.device | None = None,
        name: str | None = None,
    ):
        """Build the state body, post-concat merged stack, and Q output.

        Args:
            env: Environment wrapper providing a Box action space.
            layer_config: Hidden-layer config applied to state features.
            merged_config: Layer config applied after concatenating the action.
            output_config: Init specs for the scalar Q output layer.
            optimizer_params: Optional per-head optimizer spec.
            lr_scheduler: Optional per-head ``ScheduleWrapper``.
            device: Device to place parameters on.
            name: Display name used in error messages.
        """
        self.merged_config = list(merged_config) if merged_config else []
        super().__init__(env, layer_config, output_config, optimizer_params, lr_scheduler, device, name)

    def _build_output_layer(self) -> None:
        self.merged = SubNetwork(self.merged_config, name=f'{self.name}.merged')
        self.output_layer = nn.ModuleDict({'State_Action_value': nn.LazyLinear(1)})

    def init_weights(self) -> None:
        """Initialize body, merged stack, and output layers."""
        super().init_weights()
        self.merged.init_weights()

    def forward(self, features: T.Tensor, action: T.Tensor | None = None) -> T.Tensor:
        """Predict Q(s, a) from encoded features and an action tensor.

        Args:
            features: Encoded features from roots+trunk.
            action: Continuous action to concatenate after the state body.

        Returns:
            Scalar Q-value tensor of shape ``(..., 1)``.

        Raises:
            ValueError: If ``action`` is ``None``.
        """
        if action is None:
            raise ValueError(f"{self.name}: ContinuousQHead requires an `action` tensor")
        x = self._body_forward(features)
        merged = T.cat([x, action.to(x.device)], dim=-1)
        merged, _ = self.merged(merged)
        for layer in self.output_layer.values():
            output = layer(merged)
        return output

    def get_config(self) -> dict:
        """Serialize including ``merged_config``."""
        config = super().get_config()
        config['config'].update({'merged_config': self.merged_config})
        return config


class DiscreteQHead(Head):
    """Q(s, ·) head over discrete actions (modular counterpart of ``DiscreteCritic``)."""

    def _build_output_layer(self) -> None:
        self.num_actions = self.act_space.n
        self.output_layer = nn.ModuleDict({'Q_values': nn.LazyLinear(self.num_actions)})

    def forward(self, features: T.Tensor, action: T.Tensor | None = None) -> T.Tensor:
        """Predict a Q-value vector over discrete actions.

        Args:
            features: Encoded features from roots+trunk.
            action: Unused; accepted for a uniform head call signature.

        Returns:
            Q-values of shape ``(..., num_actions)``.
        """
        x = self._body_forward(features)
        for layer in self.output_layer.values():
            output = layer(x)
        return output


HEAD_REGISTRY: Dict[str, type] = {
    'ValueHead': ValueHead,
    'StochasticDiscreteHead': StochasticDiscreteHead,
    'StochasticContinuousHead': StochasticContinuousHead,
    'DeterministicActorHead': DeterministicActorHead,
    'ContinuousQHead': ContinuousQHead,
    'DiscreteQHead': DiscreteQHead,
}


def build_head(config: dict, env: EnvWrapper) -> Head:
    """Rebuild a head from its config dict, injecting ``env``.

    Accepts both the serialized nested form ``{"type": ..., "config": {...}}``
    (emitted by ``get_config``) and the flat YAML-authoring form
    ``{"type": ..., <head kwargs inline>}``.

    Args:
        config: Head config with a ``'type'`` key naming a ``HEAD_REGISTRY``
            entry.
        env: Live environment wrapper injected into the constructor.

    Returns:
        A concrete ``Head`` instance.

    Raises:
        ValueError: If ``type`` is not in ``HEAD_REGISTRY``.
    """
    head_type = config['type']
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type!r}. Available: {sorted(HEAD_REGISTRY)}")
    inner = config.get('config')
    if inner is None:
        inner = {k: v for k, v in config.items() if k != 'type'}
    return HEAD_REGISTRY[head_type].from_config(inner, env=env)


#: legacy Model class name -> composite Head class name (config adapter).
LEGACY_MODEL_TO_HEAD_TYPE: Dict[str, str] = {
    'StochasticDiscretePolicy': 'StochasticDiscreteHead',
    'StochasticContinuousPolicy': 'StochasticContinuousHead',
    'ValueModel': 'ValueHead',
    'ActorModel': 'DeterministicActorHead',
    'ContinuousCritic': 'ContinuousQHead',
    'DiscreteCritic': 'DiscreteQHead',
}


def head_from_legacy_model_config(config: dict, env: EnvWrapper) -> Head:
    """Translate a legacy Model ``{'type','config'}`` dict into its Head.

    Accepts head configs unchanged, so callers can pass either schema.
    ``layer_config`` / ``output_config`` / ``optimizer_params`` /
    ``lr_scheduler`` / ``distribution`` / ``temperature`` / ``merged_config``
    are lifted verbatim (``Head.from_config`` drops any legacy-only keys).

    Args:
        config: Legacy model or head config with a ``'type'`` key.
        env: Live environment wrapper injected into the constructor.

    Returns:
        A concrete ``Head`` instance.

    Raises:
        ValueError: If ``type`` is neither a known head nor a mappable legacy
            model class.
    """
    model_type = config['type']
    if model_type in HEAD_REGISTRY:
        return build_head(config, env)
    if model_type not in LEGACY_MODEL_TO_HEAD_TYPE:
        raise ValueError(
            f"Cannot adapt legacy model type {model_type!r} to a head. "
            f"Known legacy types: {sorted(LEGACY_MODEL_TO_HEAD_TYPE)}"
        )
    inner = config.get('config')
    if inner is None:
        inner = {k: v for k, v in config.items() if k != 'type'}
    return build_head({'type': LEGACY_MODEL_TO_HEAD_TYPE[model_type], 'config': inner}, env)


def modular_parts_from_config(config: dict, env: EnvWrapper) -> dict:
    """Decompose a ModularModel inner ``config`` into agent-constructor parts.

    Returns live objects (``SubNetwork`` / ``Head`` / ``ScheduleWrapper``)
    ready to pass into an Agent constructor.

    Args:
        config: Inner ``ModularModel`` config (the ``'config'`` payload of
            ``get_config``).
        env: Live environment wrapper used to build branch heads.

    Returns:
        Dict with keys ``roots``, ``trunk``, ``branches``,
        ``optimizer_params``, ``lr_scheduler``, ``shared_update``, and
        ``device``.
    """
    cfg = dict(config)
    roots = None
    if cfg.get('roots'):
        roots = {
            name: ModularModel._subnet_from_config(root_cfg, name)
            for name, root_cfg in cfg['roots'].items()
        }
    trunk = ModularModel._subnet_from_config(cfg.get('trunk'), 'trunk')
    branches = {
        role: head_from_legacy_model_config(head_cfg, env)
        for role, head_cfg in (cfg.get('branches') or {}).items()
    }
    lr_sched = cfg.get('lr_scheduler')
    if isinstance(lr_sched, dict):
        lr_sched = ScheduleWrapper.from_config(lr_sched)
    return {
        'roots': roots,
        'trunk': trunk,
        'branches': branches,
        'optimizer_params': cfg.get('optimizer_params'),
        'lr_scheduler': lr_sched,
        'shared_update': cfg.get('shared_update'),
        'device': cfg.get('device'),
    }


def map_legacy_state_dict(legacy_state: Dict[str, T.Tensor], role: str) -> Dict[str, T.Tensor]:
    """Map a legacy ``Model`` ``state_dict`` onto composite ModularModel keys.

    Legacy models store ``layers.*`` (hidden stack), optional
    ``merged_layers.*`` (``ContinuousCritic``) and ``output_layer.*``. In a
    branches-only ``ModularModel`` the same tensors live under
    ``branches.<role>.body.layers.*`` / ``branches.<role>.merged.layers.*`` /
    ``branches.<role>.output_layer.*``. Used by the old-checkpoint loader
    shim and the golden-equivalence tests.

    Args:
        legacy_state: ``state_dict`` from a legacy ``Model`` instance.
        role: Branch role name under which tensors are remapped
            (e.g. ``'policy'``, ``'critic'``).

    Returns:
        Remapped state dict keyed for a branches-only ``ModularModel``.
    """
    mapped: Dict[str, T.Tensor] = {}
    for key, value in legacy_state.items():
        if key.startswith('layers.'):
            mapped[f'branches.{role}.body.{key}'] = value
        elif key.startswith('merged_layers.'):
            mapped[f'branches.{role}.merged.layers.{key[len("merged_layers."):]}'] = value
        elif key.startswith('output_layer.'):
            mapped[f'branches.{role}.{key}'] = value
        else:
            mapped[f'branches.{role}.{key}'] = value
    return mapped


def select_policy_head(env: EnvWrapper) -> type:
    """Return the stochastic policy head class for ``env``'s action space.

    Args:
        env: Environment wrapper whose ``action_space`` is inspected.

    Returns:
        ``StochasticDiscreteHead`` for Discrete/MultiDiscrete, or
        ``StochasticContinuousHead`` for Box.

    Raises:
        ValueError: If the action space type is unsupported.
    """
    if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return StochasticDiscreteHead
    if isinstance(env.action_space, gym.spaces.Box):
        return StochasticContinuousHead
    raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")


def select_critic_head(env: EnvWrapper) -> type:
    """Return the Q-head class for ``env``'s action space.

    Args:
        env: Environment wrapper whose ``action_space`` is inspected.

    Returns:
        ``DiscreteQHead`` for Discrete/MultiDiscrete, or ``ContinuousQHead``
        for Box.

    Raises:
        ValueError: If the action space type is unsupported.
    """
    if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return DiscreteQHead
    if isinstance(env.action_space, gym.spaces.Box):
        return ContinuousQHead
    raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")


class ModularModel(nn.Module):
    """Composite roots → trunk → branches network with per-module optimizers.

    Structure (module names appear verbatim in parameter names, e.g.
    ``roots.camera.layers.conv2d_0.weight`` / ``branches.policy.body.layers...``):

    - ``roots``: optional ``{name: SubNetwork}`` — one encoder per input
      modality. Each root's flattened output is concatenated (in declaration
      order) and fed to the trunk.
    - ``trunk``: optional shared ``SubNetwork`` (identity when ``None``).
      The only module allowed to contain temporal layers (LSTM / GRU / causal
      transformer).
    - ``branches``: ``{role: Head}`` — per-role output heads (``policy``,
      ``value``, ``critic``, ``critic_b``, …).

    Gradient-ownership contract:

    - Every module with parameters gets its own optimizer over exactly its
      parameters (disjoint by construction).
    - On-policy agents combine losses into one backward and then ``step()``
      every optimizer once (equivalent to a single optimizer with param groups).
    - Off-policy agents update roots+trunk with the critic loss only; the
      policy pass uses ``detach_shared=True`` and steps only its own branch.
    """

    def __init__(
        self,
        env: EnvWrapper,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        branches: Dict[str, Head] | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str = 'combined',
        device: str | T.device | None = None,
        name: str | None = None,
    ):
        """Wire roots, trunk, and branches; dry-run; build per-module optimizers.

        Args:
            env: Environment wrapper providing observation and action spaces.
            roots: Optional named modality encoders; temporal layers forbidden.
            trunk: Optional shared body; the only place temporal layers may live.
            branches: Non-empty mapping of role name to ``Head``.
            optimizer_params: Default optimizer spec for modules that omit their
                own; defaults to Adam with ``lr=0.001``.
            lr_scheduler: Default ``ScheduleWrapper`` cloned per module that
                has no scheduler of its own.
            shared_update: Ownership hint for agents — ``'combined'``,
                ``'critic'``, or ``'policy'``.
            device: Device for parameters (``None`` = framework default).
            name: Display name used in error messages.

        Raises:
            ValueError: If ``branches`` is empty, ``shared_update`` is invalid,
                or a root/branch contains temporal layers.
        """
        super().__init__()
        if not branches:
            raise ValueError("ModularModel requires at least one branch head")
        if shared_update not in ('combined', 'critic', 'policy'):
            raise ValueError(f"Invalid shared_update: {shared_update!r}")

        self.env = env
        self.name = name or self.__class__.__name__
        self.optimizer_params = optimizer_params or {'type': 'Adam', 'params': {'lr': 0.001}}
        self._default_lr_scheduler = lr_scheduler
        self.shared_update = shared_update
        self.device = get_device(device)
        self.logger = get_logger(self.__class__.__name__, level='INFO')

        self.obs_space = (env.single_observation_space
                          if hasattr(env, 'single_observation_space')
                          else env.observation_space)
        self.act_space = (env.single_action_space
                          if hasattr(env, 'single_action_space')
                          else env.action_space)

        self.roots = nn.ModuleDict(roots) if roots else None
        self.trunk = trunk
        self.branches = nn.ModuleDict(branches)

        # Temporal layers may only live in the trunk (single memory site).
        if self.roots is not None:
            for root_name, root in self.roots.items():
                if root.is_temporal:
                    raise ValueError(
                        f"Root '{root_name}' contains temporal layers; lstm/gru/causal "
                        f"transformer_encoder are only allowed in the trunk."
                    )
        for role, head in self.branches.items():
            if head.body.is_temporal or (isinstance(head, ContinuousQHead) and head.merged.is_temporal):
                raise ValueError(
                    f"Branch '{role}' contains temporal layers; lstm/gru/causal "
                    f"transformer_encoder are only allowed in the trunk."
                )

        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.lr_schedulers: Dict[str, ScheduleWrapper] = {}

        self.to(self.device)
        self._init_model()

    # ------------------------------------------------------------------ #
    # module bookkeeping
    # ------------------------------------------------------------------ #
    def module_map(self) -> Dict[str, nn.Module]:
        """Ordered map of ``roots.<name>`` / ``trunk`` / ``branches.<role>`` modules.

        Returns:
            Mapping from canonical module name to the live ``nn.Module``.
        """
        mods: Dict[str, nn.Module] = {}
        if self.roots is not None:
            for root_name, root in self.roots.items():
                mods[f'roots.{root_name}'] = root
        if self.trunk is not None:
            mods['trunk'] = self.trunk
        for role, head in self.branches.items():
            mods[f'branches.{role}'] = head
        return mods

    def shared_module_names(self) -> List[str]:
        """Names of roots/trunk modules that own parameters (have optimizers)."""
        return [n for n in self.optimizers if n == 'trunk' or n.startswith('roots.')]

    def branch_module_names(self, *roles: str) -> List[str]:
        """Optimizer names for the given branch roles (all branches if empty).

        Args:
            *roles: Branch role names to select; omit to include every branch
                that has an optimizer.

        Returns:
            Optimizer keys of the form ``branches.<role>``.

        Raises:
            KeyError: If a requested role is not in ``branches``.
        """
        if not roles:
            return [n for n in self.optimizers if n.startswith('branches.')]
        names = []
        for role in roles:
            if role not in self.branches:
                raise KeyError(f"Unknown branch role: {role!r}. Available: {list(self.branches.keys())}")
            full = f'branches.{role}'
            if full in self.optimizers:
                names.append(full)
        return names

    @property
    def is_recurrent(self) -> bool:
        """True when the trunk contains recurrent layers."""
        return self.trunk is not None and self.trunk.is_recurrent

    @property
    def is_causal(self) -> bool:
        """True when the trunk contains a causal transformer block."""
        return self.trunk is not None and self.trunk.is_causal

    @property
    def is_temporal(self) -> bool:
        """True when the trunk is recurrent or causal."""
        return self.is_recurrent or self.is_causal

    def init_hidden(self, batch_size: int) -> Dict[str, Any]:
        """Allocate zero recurrent states keyed ``trunk.<layer_key>``.

        Args:
            batch_size: Leading batch size for the states.

        Returns:
            Hidden-state dict (empty when the trunk is non-recurrent).
        """
        hidden: Dict[str, Any] = {}
        if self.trunk is not None:
            for k, v in self.trunk.init_hidden(batch_size, self.device).items():
                hidden[f'trunk.{k}'] = v
        return hidden

    @staticmethod
    def detach_hidden(hidden: Dict[str, Any] | None) -> Dict[str, Any]:
        """Return a detached copy of a hidden-state dict.

        Args:
            hidden: Hidden-state dict (or ``None`` for empty).

        Returns:
            Detached copy safe to carry across optimizer steps.
        """
        out: Dict[str, Any] = {}
        for k, v in (hidden or {}).items():
            if isinstance(v, tuple):
                out[k] = tuple(t.detach() for t in v)
            else:
                out[k] = v.detach()
        return out

    @staticmethod
    def index_hidden(hidden: Dict[str, Any] | None, idx) -> Dict[str, Any]:
        """Select a batch subset of a hidden-state dict.

        The batch dimension is dim 1: recurrent states are shaped
        ``(num_layers, B, H)``.

        Args:
            hidden: Hidden-state dict (or ``None`` for empty).
            idx (int | slice | torch.Tensor): Batch indices accepted by tensor
                indexing on dim 1.

        Returns:
            Indexed hidden-state dict with the same keys.
        """
        out: Dict[str, Any] = {}
        for k, v in (hidden or {}).items():
            if isinstance(v, tuple):
                out[k] = tuple(t[:, idx].contiguous() for t in v)
            else:
                out[k] = v[:, idx].contiguous()
        return out

    def mask_hidden(self, hidden: Dict[str, Any] | None, start_mask: T.Tensor) -> Dict[str, Any]:
        """Zero hidden entries for envs flagged in ``start_mask`` (episode starts).

        Args:
            hidden: Hidden-state dict (or ``None`` / empty for no-op).
            start_mask: Bool tensor of episode-start flags, shape ``(B,)``.

        Returns:
            Masked hidden-state dict (or ``{}`` when there is nothing to mask).
        """
        if not hidden or self.trunk is None:
            return hidden or {}
        keep = ~start_mask.bool()
        out: Dict[str, Any] = {}
        for key, value in hidden.items():
            layer_key = key.split('.', 1)[1]
            layer = self.trunk.layers[layer_key]
            out[key] = layer.mask_hidden(value, keep)
        return out

    def forward_context(
        self,
        obs_window,
        action: T.Tensor | None = None,
        goal: T.Tensor | None = None,
        branches=None,
        start_mask: T.Tensor | None = None,
    ) -> Dict[str, Any]:
        """Run heads on the last step of a causal observation window.

        Encodes a ``(B, W, ...)`` observation window in sequence mode and
        evaluates the requested heads on the last time position only
        (rolling-window rollout for causal-transformer trunks).

        Args:
            obs_window (torch.Tensor | dict): Observation window with a leading
                ``(B, W, ...)`` layout (dict values share that layout).
            action: Optional action consumed by ContinuousQHead branches.
            goal: Optional goal tensor (same routing rules as ``forward``).
            branches (str | list | tuple | None): Role name, sequence of roles,
                or ``None`` for all.
            start_mask: Optional ``(B, W)`` episode-start flags.

        Returns:
            Mapping of role name to that head's output at the last position.
        """
        if isinstance(branches, str):
            branch_roles = [branches]
        elif branches is None:
            branch_roles = list(self.branches.keys())
        else:
            branch_roles = list(branches)
        features, _ = self._encode(obs_window, goal=goal, start_mask=start_mask,
                                   mode='sequence')
        last = features[:, -1]
        if action is not None:
            action = action.to(self.device)
        return {role: self.branches[role](last, action=action) for role in branch_roles}

    @staticmethod
    def hidden_to_tensors(hidden: Dict[str, Any] | None) -> Dict[str, T.Tensor]:
        """Flatten a hidden dict to batch-first plain tensors for storage.

        LSTM tuples split into ``<key>.h`` / ``<key>.c``; GRU states become
        ``<key>.g``. Each ``(num_layers, B, H)`` state is transposed to
        ``(B, num_layers, H)`` so storage rings/buffers see a leading batch dim.

        Args:
            hidden: Hidden-state dict (or ``None`` for empty).

        Returns:
            Flat mapping of suffix-tagged keys to batch-first tensors.
        """
        flat: Dict[str, T.Tensor] = {}
        for k, v in (hidden or {}).items():
            if isinstance(v, tuple):
                flat[f'{k}.h'] = v[0].detach().transpose(0, 1).contiguous()
                flat[f'{k}.c'] = v[1].detach().transpose(0, 1).contiguous()
            else:
                flat[f'{k}.g'] = v.detach().transpose(0, 1).contiguous()
        return flat

    @staticmethod
    def hidden_from_tensors(flat: Dict[str, T.Tensor] | None) -> Dict[str, Any]:
        """Inverse of ``hidden_to_tensors`` (back to ``(layers, B, H)``).

        Args:
            flat: Flat storage dict produced by ``hidden_to_tensors``.

        Returns:
            Nested hidden-state dict with LSTM tuples restored.
        """
        hidden: Dict[str, Any] = {}
        for k, v in (flat or {}).items():
            base, suffix = k.rsplit('.', 1)
            if suffix == 'g':
                hidden[base] = v.transpose(0, 1).contiguous()
            elif suffix == 'h':
                c = flat[f'{base}.c']
                hidden[base] = (v.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous())
        return hidden

    # ------------------------------------------------------------------ #
    # optimizer coordination
    # ------------------------------------------------------------------ #
    def _resolve_optimizer_names(self, modules: List[str] | str | None = None) -> List[str]:
        if modules is None:
            return list(self.optimizers.keys())
        if isinstance(modules, str):
            modules = [modules]
        known = self.module_map()
        names = []
        for m in modules:
            if m in self.optimizers:
                names.append(m)
            elif m in known:
                continue  # module exists but has no trainable params
            else:
                raise KeyError(f"Unknown module {m!r}. Available: {list(known.keys())}")
        return names

    def zero_grad(self, modules: List[str] | str | None = None) -> None:
        """Zero gradients for the selected modules' optimizers.

        Args:
            modules: Module name, list of names, or ``None`` for every
                optimizer. Names match ``module_map`` keys.
        """
        for module_name in self._resolve_optimizer_names(modules):
            self.optimizers[module_name].zero_grad()

    def step(self, modules: List[str] | str | None = None) -> None:
        """Step the selected modules' optimizers.

        Args:
            modules: Module name, list of names, or ``None`` for every
                optimizer. Names match ``module_map`` keys.
        """
        for module_name in self._resolve_optimizer_names(modules):
            self.optimizers[module_name].step()

    def clip(self, max_norm: float, modules: List[str] | str | None = None) -> float:
        """Clip the grad norm over the union of the given modules' parameters.

        Args:
            max_norm: Maximum allowed total norm.
            modules: Module name, list of names, or ``None`` for every
                optimizer.

        Returns:
            Pre-clip total norm (``0.0`` when there are no parameters).
        """
        params = []
        for module_name in self._resolve_optimizer_names(modules):
            for group in self.optimizers[module_name].param_groups:
                params.extend(group['params'])
        if not params:
            return 0.0
        return float(T.nn.utils.clip_grad_norm_(params, max_norm=max_norm))

    def learning_rate(self, module: str) -> float:
        """Current LR of a module's optimizer (``0.0`` when it has none).

        Args:
            module: Canonical module name (a ``module_map`` key).

        Returns:
            Learning rate from the first param group, or ``0.0``.
        """
        opt = self.optimizers.get(module)
        return float(opt.param_groups[0]['lr']) if opt is not None else 0.0

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        obs: T.Tensor | Dict[str, T.Tensor],
        action: T.Tensor | None = None,
        goal: T.Tensor | None = None,
        branches: List[str] | tuple | str | None = None,
        hidden: Dict[str, Any] | None = None,
        start_mask: T.Tensor | None = None,
        detach_shared: bool = False,
        mode: str = 'step',
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Encode observations through roots+trunk and run the requested heads.

        Args:
            obs: Tensor (flat obs) or dict of tensors (multi-modal obs).
                Step mode shapes are ``(B, ...)``; sequence mode ``(B, T, ...)``.
            action: Action tensor consumed by ContinuousQHead branches.
            goal: Optional goal tensor. For flat obs it is concatenated to the
                input (legacy behavior); for dict obs it is injected under the
                reserved input key ``'goal'`` for roots that request it.
            branches: Role name(s) to run (default: all branches).
            hidden: Recurrent state dict from a previous call (zeros if None).
            start_mask: Episode-start flags — ``(B,)`` step / ``(B, T)`` sequence.
            detach_shared: Detach the shared (roots+trunk) features so
                gradients from the requested heads never reach shared modules.
            mode: ``'step'`` or ``'sequence'``.

        Returns:
            outputs (dict): Mapping of role name to that head's output.
            new_hidden (dict): Updated recurrent states keyed ``trunk.<layer>``.

        Raises:
            ValueError: If ``mode`` is invalid.
            KeyError: If a requested branch role is unknown.
            RuntimeError: If a branch forward fails.
        """
        if mode not in ('step', 'sequence'):
            raise ValueError(f"Invalid mode: {mode!r}")
        if isinstance(branches, str):
            branch_roles = [branches]
        elif branches is None:
            branch_roles = list(self.branches.keys())
        else:
            branch_roles = list(branches)
        for role in branch_roles:
            if role not in self.branches:
                raise KeyError(f"Unknown branch role: {role!r}. Available: {list(self.branches.keys())}")

        features, new_hidden = self._encode(obs, goal=goal, hidden=hidden,
                                            start_mask=start_mask, mode=mode)
        if detach_shared:
            features = features.detach()

        if action is not None:
            action = action.to(self.device)

        outputs: Dict[str, Any] = {}
        for role in branch_roles:
            try:
                outputs[role] = self.branches[role](features, action=action)
            except Exception as e:
                raise RuntimeError(
                    f"{self.name}: branch '{role}' forward failed "
                    f"(features shape {tuple(features.shape)}): {e}"
                ) from e
        return outputs, new_hidden

    def _preprocess_input(self, x: T.Tensor, subnet: SubNetwork | None = None) -> T.Tensor:
        """Per-input preprocessing: device/dtype and shape normalization.

        Extends the legacy ``Model._preprocess_state`` with uint8 image
        scaling (cast to float and divide by 255, SB3 convention). The
        image-shape heuristics (grayscale channel dim, HWC->CHW permute) are
        applied for the legacy flat path (``subnet is None``) and for roots
        whose stacks contain convolutional layers; token (embedding) roots
        keep integer dtype.
        """
        x = x.to(self.device)

        if subnet is not None and subnet.expects_tokens:
            if not x.dtype == T.long:
                x = x.long()
            return x

        if x.dtype == T.uint8:
            x = x.float() / 255.0
        elif x.dtype != T.float32:
            x = x.float()

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if subnet is None:
            # Legacy flat path: exact parity with Model._preprocess_state
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if isinstance(self.env, GymnasiumWrapper):
                if x.dim() == 4 and x.shape[-1] in [3, 4]:
                    x = x.permute(0, 3, 1, 2)
        elif subnet.expects_image:
            if x.dim() == 3:
                x = x.unsqueeze(1)
            # Channels-last -> channels-first for ANY env source: IsaacLab
            # cameras (TiledCamera/Camera) and Gymnasium image envs both emit
            # (N, H, W, C). Skipped when the input is already channels-first
            # (dim 1 channel-like), which real camera frames never are.
            if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
                x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _run_root(self, root_name: str, root: SubNetwork,
                  inputs: T.Tensor | Dict[str, T.Tensor],
                  goal: T.Tensor | None, mode: str) -> T.Tensor:
        """Assemble a root's input, run it, and flatten to (B, D) / (B, T, D)."""
        if isinstance(inputs, dict):
            if not root.input_keys:
                raise ValueError(
                    f"Root '{root_name}' must declare input_keys to consume dict observations "
                    f"(available keys: {list(inputs.keys())})"
                )
            parts = []
            for key in root.input_keys:
                if key == GOAL_INPUT_KEY and key not in inputs:
                    if goal is None:
                        raise ValueError(
                            f"Root '{root_name}' requests the 'goal' input but no goal was passed"
                        )
                    part = goal
                elif key in inputs:
                    part = inputs[key]
                else:
                    raise KeyError(
                        f"Root '{root_name}' input key {key!r} not found in observation "
                        f"(available: {list(inputs.keys())})"
                    )
                parts.append(part)
        else:
            if root.input_keys:
                raise ValueError(
                    f"Root '{root_name}' declares input_keys {root.input_keys} but received "
                    f"a flat (non-dict) observation"
                )
            parts = [inputs]

        if mode == 'sequence':
            # Fold time into batch for the (possibly convolutional) root.
            batch_size, seq_len = parts[0].shape[0], parts[0].shape[1]
            folded = [p.reshape(batch_size * seq_len, *p.shape[2:]) for p in parts]
            processed = [self._preprocess_input(p, root) for p in folded]
            x = processed[0] if len(processed) == 1 else T.cat(processed, dim=-1)
            try:
                y, _ = root(x, mode='step')
            except Exception as e:
                raise RuntimeError(
                    f"{self.name}: root '{root_name}' forward failed "
                    f"(input shape {tuple(x.shape)}): {e}"
                ) from e
            y = y.reshape(y.shape[0], -1)  # flatten
            return y.reshape(batch_size, seq_len, -1)

        processed = [self._preprocess_input(p, root) for p in parts]
        x = processed[0] if len(processed) == 1 else T.cat(processed, dim=-1)
        try:
            y, _ = root(x, mode='step')
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: root '{root_name}' forward failed "
                f"(input shape {tuple(x.shape)}): {e}"
            ) from e
        return y.reshape(y.shape[0], -1)  # flatten

    def _encode(
        self,
        obs: T.Tensor | Dict[str, T.Tensor],
        goal: T.Tensor | None = None,
        hidden: Dict[str, Any] | None = None,
        start_mask: T.Tensor | None = None,
        mode: str = 'step',
    ) -> tuple[T.Tensor, Dict[str, Any]]:
        """Roots -> concat -> trunk. Returns (features, new_hidden)."""
        if goal is not None:
            goal = goal.to(self.device)

        new_hidden: Dict[str, Any] = {}

        if self.roots is not None:
            feats = [
                self._run_root(root_name, root, obs, goal, mode)
                for root_name, root in self.roots.items()
            ]
            features = feats[0] if len(feats) == 1 else T.cat(feats, dim=-1)
            # When goal is passed but not consumed by any root via input_keys,
            # concatenate it to the fused features (legacy-style conditioning).
            if goal is not None and not any(
                root.input_keys and GOAL_INPUT_KEY in root.input_keys
                for root in self.roots.values()
            ):
                features = T.cat([features, goal], dim=-1)
        else:
            if isinstance(obs, dict):
                raise ValueError(
                    "Dict observations require roots with input_keys "
                    f"(got keys: {list(obs.keys())})"
                )
            if mode == 'sequence':
                batch_size, seq_len = obs.shape[0], obs.shape[1]
                folded = obs.reshape(batch_size * seq_len, *obs.shape[2:])
                features = self._preprocess_input(folded)
                features = features.reshape(batch_size, seq_len, *features.shape[1:])
            else:
                features = self._preprocess_input(obs)
            if goal is not None:
                features = T.cat([features, goal], dim=-1)

        if self.trunk is not None:
            sub_hidden = None
            if hidden:
                prefix = 'trunk.'
                sub_hidden = {
                    k[len(prefix):]: v for k, v in hidden.items() if k.startswith(prefix)
                } or None
            try:
                features, trunk_hidden = self.trunk(features, sub_hidden, start_mask, mode)
            except Exception as e:
                raise RuntimeError(
                    f"{self.name}: trunk forward failed "
                    f"(input shape {tuple(features.shape)}): {e}"
                ) from e
            for k, v in trunk_hidden.items():
                new_hidden[f'trunk.{k}'] = v

        return features, new_hidden

    # ------------------------------------------------------------------ #
    # initialization (dry run -> weights -> optimizers)
    # ------------------------------------------------------------------ #
    def _build_dummy_inputs(self, batch_shape: tuple) -> tuple:
        """(obs, action, goal) dummy tensors with the given batch prefix."""
        obs_space = self.obs_space
        is_dict_space = isinstance(obs_space, gym.spaces.Dict)

        action_dummy = None
        if any(isinstance(h, ContinuousQHead) for h in self.branches.values()):
            action_dummy = T.ones((*batch_shape, *self.act_space.shape),
                                  device=self.device, dtype=T.float)

        goal_dummy = None
        goal_key = getattr(self.env, 'goal_key', None)

        uses_dict_routing = self.roots is not None and any(
            root.input_keys for root in self.roots.values()
        )
        if uses_dict_routing:
            if not is_dict_space:
                raise ValueError(
                    "Roots declare input_keys but the observation space is not a Dict space"
                )
            obs_dummy: Dict[str, T.Tensor] = {}
            wants_goal = False
            for root in self.roots.values():
                for key in (root.input_keys or []):
                    if key == GOAL_INPUT_KEY:
                        wants_goal = True
                        continue
                    if key not in obs_space.spaces:
                        raise ValueError(
                            f"Root input key {key!r} not in observation space "
                            f"(available: {list(obs_space.spaces.keys())})"
                        )
                    space = obs_space.spaces[key]
                    obs_dummy[key] = T.ones((*batch_shape, *space.shape),
                                            device=self.device, dtype=T.float)
            if wants_goal or (goal_key is not None and goal_key in getattr(obs_space, 'spaces', {})):
                if goal_key is None or goal_key not in obs_space.spaces:
                    raise ValueError(
                        "A root requests the 'goal' input but the env has no goal_key space"
                    )
                goal_shape = obs_space.spaces[goal_key].shape
                goal_dummy = T.ones((*batch_shape, *goal_shape), device=self.device, dtype=T.float)
            return obs_dummy, action_dummy, goal_dummy

        # Flat-input path (mirrors legacy Model._init_model)
        obs_key = getattr(self.env, 'obs_key', None)
        is_goal_conditioned = (
            is_dict_space
            and obs_key in getattr(obs_space, 'spaces', {})
            and goal_key is not None
            and goal_key in obs_space.spaces
        )
        if is_dict_space:
            if obs_key in obs_space.spaces:
                obs_shape = obs_space.spaces[obs_key].shape
            else:
                raise ValueError(
                    f"Dict observation space requires env.obs_key to select the input "
                    f"(obs_key={obs_key!r}, available: {list(obs_space.spaces.keys())})"
                )
        else:
            obs_shape = obs_space.shape
        obs_dummy = T.ones((*batch_shape, *obs_shape), device=self.device, dtype=T.float)
        if is_goal_conditioned:
            goal_shape = obs_space.spaces[goal_key].shape
            goal_dummy = T.ones((*batch_shape, *goal_shape), device=self.device, dtype=T.float)
        return obs_dummy, action_dummy, goal_dummy

    def _init_model(self) -> None:
        # 1. Step-mode dry run: materializes every lazy module.
        obs_dummy, action_dummy, goal_dummy = self._build_dummy_inputs((32,))
        with T.no_grad():
            self.forward(obs_dummy, action=action_dummy, goal=goal_dummy, mode='step')

        # 2. Sequence-mode dry run (validation) when the trunk is temporal.
        if self.is_temporal:
            seq_obs, seq_action, seq_goal = self._build_dummy_inputs((4, 3))
            with T.no_grad():
                self.forward(seq_obs, action=seq_action, goal=seq_goal, mode='sequence')

        # 3. Weight initialization per layer_config/output_config kernels.
        for module in self.module_map().values():
            module.init_weights()

        # 4. Per-module optimizers over disjoint parameter sets.
        for module_name, module in self.module_map().items():
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                continue
            opt_spec = getattr(module, 'optimizer_params', None) or self.optimizer_params
            self.optimizers[module_name] = build_optimizer(params, opt_spec)
            scheduler = getattr(module, 'lr_scheduler', None)
            if scheduler is None and self._default_lr_scheduler is not None:
                scheduler = self._default_lr_scheduler.clone()
            if scheduler is not None:
                scheduler.attach_optimizer(self.optimizers[module_name])
                self.lr_schedulers[module_name] = scheduler

    # ------------------------------------------------------------------ #
    # serialization / cloning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _subnet_config(subnet: SubNetwork | None) -> dict | None:
        if subnet is None:
            return None
        return {
            'layer_config': subnet.layer_config,
            'input_keys': subnet.input_keys,
            'optimizer_params': subnet.optimizer_params,
            'lr_scheduler': subnet.lr_scheduler.get_config() if subnet.lr_scheduler else None,
        }

    @staticmethod
    def _subnet_from_config(cfg: dict | None, name: str) -> SubNetwork | None:
        if cfg is None:
            return None
        lr_sched = cfg.get('lr_scheduler')
        if isinstance(lr_sched, dict):
            lr_sched = ScheduleWrapper.from_config(lr_sched)
        return SubNetwork(
            layer_config=cfg.get('layer_config'),
            input_keys=cfg.get('input_keys'),
            optimizer_params=cfg.get('optimizer_params'),
            lr_scheduler=lr_sched,
            name=name,
        )

    def get_config(self) -> dict:
        """Serialize architecture to a ``{'type', 'config'}`` dict."""
        return {
            'type': 'ModularModel',
            'config': {
                'roots': (
                    {name: self._subnet_config(root) for name, root in self.roots.items()}
                    if self.roots is not None else None
                ),
                'trunk': self._subnet_config(self.trunk),
                'branches': {role: head.get_config() for role, head in self.branches.items()},
                'optimizer_params': self.optimizer_params,
                'lr_scheduler': (self._default_lr_scheduler.get_config()
                                 if self._default_lr_scheduler else None),
                'shared_update': self.shared_update,
                'device': self.device.type,
            },
        }

    @classmethod
    def from_config(cls, config: dict, env: EnvWrapper) -> 'ModularModel':
        """Rebuild a ModularModel from an inner config dict, injecting ``env``.

        Args:
            config: Inner config (the ``'config'`` payload of ``get_config``).
            env: Live environment wrapper injected into branch constructors.

        Returns:
            A new ``ModularModel`` with fresh weights and optimizers.
        """
        cfg = dict(config)
        roots = None
        if cfg.get('roots'):
            roots = {
                name: cls._subnet_from_config(root_cfg, name)
                for name, root_cfg in cfg['roots'].items()
            }
        trunk = cls._subnet_from_config(cfg.get('trunk'), 'trunk')
        branches = {
            role: build_head(head_cfg, env)
            for role, head_cfg in cfg['branches'].items()
        }
        lr_sched = cfg.get('lr_scheduler')
        if isinstance(lr_sched, dict):
            lr_sched = ScheduleWrapper.from_config(lr_sched)
        return cls(
            env=env,
            roots=roots,
            trunk=trunk,
            branches=branches,
            optimizer_params=cfg.get('optimizer_params'),
            lr_scheduler=lr_sched,
            shared_update=cfg.get('shared_update', 'combined'),
            device=cfg.get('device'),
        )

    def save_state(self, path: Path | str) -> None:
        """Write weights plus per-module optimizer/scheduler state to one ``.pt``.

        Args:
            path: Destination file path (parent directories are created).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model': self.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'lr_schedulers': {name: sched.get_state() for name, sched in self.lr_schedulers.items()},
        }
        temperature_schedules = {}
        for role, head in self.branches.items():
            ts = getattr(head, 'temperature_schedule', None)
            if ts is not None:
                temperature_schedules[role] = ts.get_state()
        if temperature_schedules:
            state['temperature_schedules'] = temperature_schedules
        T.save(state, path)

    def load_state(self, path: Path | str, load_weights: bool = True) -> None:
        """Restore state written by ``save_state`` (in place).

        Args:
            path: Path to a ``.pt`` file produced by ``save_state``.
            load_weights: If False, restore optimizers/schedulers only.
        """
        state = T.load(Path(path), map_location=self.device, weights_only=False)
        if load_weights and state.get('model') is not None:
            self.load_state_dict(state['model'])
        for name, opt_state in (state.get('optimizers') or {}).items():
            if name in self.optimizers and opt_state is not None:
                self.optimizers[name].load_state_dict(opt_state)
        for name, sched_state in (state.get('lr_schedulers') or {}).items():
            if name in self.lr_schedulers and sched_state is not None:
                self.lr_schedulers[name].set_state(sched_state)
        for role, ts_state in (state.get('temperature_schedules') or {}).items():
            head = self.branches[role] if role in self.branches else None
            ts = getattr(head, 'temperature_schedule', None) if head is not None else None
            if ts is not None and ts_state is not None:
                ts.set_state(ts_state)

    def clone(
        self,
        copy_weights: bool = True,
        branches: List[str] | None = None,
        device: Optional[str | T.device] = None,
    ) -> 'ModularModel':
        """Clone the composite (optionally a branch subset, e.g. for targets).

        The live env instance is reused (models only read spaces/keys from it).

        Args:
            copy_weights: If True, copy matching parameters into the clone.
            branches: Optional subset of branch roles to keep; ``None`` keeps
                all.
            device: Device for the clone (defaults to this model's device).

        Returns:
            A new ``ModularModel`` sharing the same env.

        Raises:
            KeyError: If a requested branch role is unknown.
        """
        device = get_device(device) if device is not None else self.device
        cfg = self.get_config()['config']
        if branches is not None:
            missing = [r for r in branches if r not in cfg['branches']]
            if missing:
                raise KeyError(f"Unknown branch role(s) for clone: {missing}")
            cfg['branches'] = {r: c for r, c in cfg['branches'].items() if r in branches}
        cfg['device'] = device.type
        cloned = ModularModel.from_config(cfg, env=self.env)
        if copy_weights:
            own_state = self.state_dict()
            subset = {k: v for k, v in own_state.items() if k in cloned.state_dict()}
            cloned.load_state_dict(subset, strict=True)
        return cloned

    def set_device(self, device: str | T.device) -> 'ModularModel':
        """Move the composite to ``device`` and update internal device attrs.

        Moves parameters, buffers, head-held tensors (e.g. action bounds), and
        optimizer state tensors.

        Args:
            device: Target device (resolved via ``get_device``).

        Returns:
            ``self``, for chaining.
        """
        device = get_device(device)
        self.device = device
        for head in self.branches.values():
            head.device = device
            for attr in ('act_space_low', 'act_space_high'):
                tensor = getattr(head, attr, None)
                if isinstance(tensor, T.Tensor):
                    setattr(head, attr, tensor.to(device))
        self.to(device)
        for opt in self.optimizers.values():
            for state in opt.state.values():
                for key, value in state.items():
                    if isinstance(value, T.Tensor):
                        state[key] = value.to(device)
        return self


# Registry of every concrete model class, keyed by class name (the "type" tag
# emitted by get_config). Used by build_model to reconstruct from a config.
MODEL_REGISTRY: Dict[str, type] = {
    "StochasticDiscretePolicy": StochasticDiscretePolicy,
    "StochasticContinuousPolicy": StochasticContinuousPolicy,
    "ValueModel": ValueModel,
    "ActorModel": ActorModel,
    "ContinuousCritic": ContinuousCritic,
    "DiscreteCritic": DiscreteCritic,
    "ModularModel": ModularModel,
}


def build_model(config: dict, env: EnvWrapper) -> Model:
    """Rebuild a model from a ``{"type", "config"}`` dict, injecting ``env``."""
    model_type = config["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type!r}")
    return MODEL_REGISTRY[model_type].from_config(config["config"], env=env)


def build_layers(types: List[str], units_per_layer: List[int], initializers: List[str], kernel_params:List[dict]):
    """Zip parallel layer lists into a legacy ``layer_config`` list.

    Args:
        types: Layer type names (e.g. ``'dense'``).
        units_per_layer: Unit counts aligned with ``types``.
        initializers: Kernel initializer names aligned with ``types``.
        kernel_params: Per-layer kernel kwargs dicts aligned with ``types``.

    Returns:
        layers (list[dict]): List of ``{'type', 'params'}`` layer specs.
    """
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
    """Return the legacy stochastic policy class for ``env``'s action space.

    Args:
        env: Environment wrapper whose ``action_space`` is inspected.

    Returns:
        model_class (type): ``StochasticDiscretePolicy`` for Discrete /
            MultiDiscrete, or ``StochasticContinuousPolicy`` for Box.

    Raises:
        ValueError: If the action space type is unsupported.
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
    """Return the legacy critic class for ``env``'s action space.

    Args:
        env: Environment wrapper whose ``action_space`` is inspected.

    Returns:
        model_class (type): ``DiscreteCritic`` for Discrete / MultiDiscrete, or
            ``ContinuousCritic`` for Box.

    Raises:
        ValueError: If the action space type is unsupported.
    """
    if isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete):
        model_class = DiscreteCritic
    elif isinstance(env.action_space, gym.spaces.Box):
        model_class = ContinuousCritic
    else:
        raise ValueError("Unsupported action space type. Only Discrete and Box spaces are supported.")
    return model_class
