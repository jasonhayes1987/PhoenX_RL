"""Parameter specs for every ``LAYER_REGISTRY`` layer type.

This module imports no torch, so the config schema tooling and GUI can read
it without loading the training stack. ``phoenx.models.build_layer`` uses it
to reject unknown layer params, and a test keeps it in lock-step with
``LAYER_REGISTRY``.

Each entry is either

* ``{"params": {name: spec, ...}}`` — the registry lambda reads exactly these
  names (often with ``p.get``, which would otherwise ignore a misspelled
  key), so any other name is rejected; or
* ``{"passthrough": "torch.nn.Xxx", "params": {...}}`` — the params dict is
  splatted into a torch constructor, which raises on unknown names itself.
  ``params`` lists the common ones for the schema / GUI, and other torch
  keyword arguments are allowed.

A ``spec`` is ``{"type": <json type>, "default": <value>}``, or
``{"type": ..., "required": True}`` for params the lambda indexes directly.

Weight-initialization keys (``kernel``, ``kernel_params``) are valid on every
layer type and are listed separately in [INIT_SPEC_KEYS][phoenx.layer_specs.INIT_SPEC_KEYS].
"""

from __future__ import annotations

#: Keys in a layer's ``params`` that configure weight init rather than the
#: layer constructor. ``'kernel params'`` (with a space) is a legacy spelling.
INIT_SPEC_KEYS = ("kernel", "kernel_params", "kernel params")

#: Accepted ``kernel`` names (see ``phoenx.models.apply_kernel_``).
KERNEL_NAMES = (
    "default", "kaiming_uniform", "kaiming_normal", "xavier_uniform",
    "xavier_normal", "truncated_normal", "uniform", "normal", "orthogonal",
    "constant", "ones", "zeros", "variance_scaling",
)

#: Layer types with rank-2+ weight tensors, the only ones ``kernel`` /
#: ``kernel_params`` affect (see ``phoenx.models.init_module_weights``).
WEIGHTED_LAYERS = frozenset({
    "dense", "linear", "conv1d", "conv2d", "conv3d", "convtranspose2d",
    "embedding", "positional_encoding", "mha", "transformer_encoder", "lstm", "gru",
})

_INT = "integer"
_NUM = "number"
_BOOL = "boolean"
_STR = "string"


def _conv(dim: str) -> dict:
    return {"params": {
        "out_channels": {"type": _INT, "default": 64},
        "kernel_size": {"type": [_INT, "array"], "default": 3},
        "stride": {"type": [_INT, "array"], "default": 1},
        "padding": {"type": [_INT, "array", _STR], "default": 0},
        "bias": {"type": _BOOL, "default": True},
    }, "doc": f"Lazy {dim} convolution (input channels inferred)."}


def _pool(cls: str) -> dict:
    return {"passthrough": f"torch.nn.{cls}", "params": {
        "kernel_size": {"type": [_INT, "array"], "required": True},
        "stride": {"type": [_INT, "array", "null"], "default": None},
        "padding": {"type": [_INT, "array"], "default": 0},
    }}


_NO_PARAMS = {"params": {}}

LAYER_SPECS: dict[str, dict] = {
    # Linear
    "dense": {"params": {"units": {"type": _INT, "required": True}},
              "doc": "Fully connected layer; input size inferred."},
    "linear": {"params": {
        "in_features": {"type": _INT, "required": True},
        "out_features": {"type": _INT, "required": True},
        "bias": {"type": _BOOL, "default": True},
    }},
    # Convolutional
    "conv1d": _conv("1D"),
    "conv2d": _conv("2D"),
    "conv3d": _conv("3D"),
    "convtranspose2d": _conv("2D transposed"),
    # Pooling
    "pool": _pool("MaxPool2d"),
    "maxpool1d": _pool("MaxPool1d"),
    "maxpool2d": _pool("MaxPool2d"),
    "avgpool1d": _pool("AvgPool1d"),
    "avgpool2d": _pool("AvgPool2d"),
    "adaptiveavgpool2d": {"params": {"output_size": {"type": [_INT, "array"], "default": 1}}},
    "adaptivemaxpool2d": {"params": {"output_size": {"type": [_INT, "array"], "default": 1}}},
    # Normalization
    "batchnorm1d": {"passthrough": "torch.nn.LazyBatchNorm1d", "params": {
        "eps": {"type": _NUM, "default": 1e-5}, "momentum": {"type": _NUM, "default": 0.1}}},
    "batchnorm2d": {"passthrough": "torch.nn.LazyBatchNorm2d", "params": {
        "eps": {"type": _NUM, "default": 1e-5}, "momentum": {"type": _NUM, "default": 0.1}}},
    "batchnorm3d": {"passthrough": "torch.nn.LazyBatchNorm3d", "params": {
        "eps": {"type": _NUM, "default": 1e-5}, "momentum": {"type": _NUM, "default": 0.1}}},
    "layernorm": {"passthrough": "torch.nn.LayerNorm", "params": {
        "normalized_shape": {"type": [_INT, "array"], "required": True},
        "eps": {"type": _NUM, "default": 1e-5}}},
    "groupnorm": {"passthrough": "torch.nn.GroupNorm", "params": {
        "num_groups": {"type": _INT, "required": True},
        "num_channels": {"type": _INT, "required": True},
        "eps": {"type": _NUM, "default": 1e-5}}},
    "rmsnorm": {"passthrough": "torch.nn.RMSNorm", "params": {
        "normalized_shape": {"type": [_INT, "array"], "required": True},
        "eps": {"type": [_NUM, "null"], "default": None}}},
    # Activations
    "relu": _NO_PARAMS,
    "leakyrelu": {"passthrough": "torch.nn.LeakyReLU", "params": {
        "negative_slope": {"type": _NUM, "default": 0.01}}},
    "tanh": _NO_PARAMS,
    "sigmoid": _NO_PARAMS,
    "gelu": {"passthrough": "torch.nn.GELU", "params": {
        "approximate": {"type": _STR, "enum": ["none", "tanh"], "default": "none"}}},
    "silu": _NO_PARAMS,
    "elu": {"passthrough": "torch.nn.ELU", "params": {"alpha": {"type": _NUM, "default": 1.0}}},
    "mish": _NO_PARAMS,
    "softmax": {"params": {"dim": {"type": _INT, "default": -1}}},
    "softplus": {"passthrough": "torch.nn.Softplus", "params": {
        "beta": {"type": _NUM, "default": 1.0}, "threshold": {"type": _NUM, "default": 20.0}}},
    # Regularization
    "dropout": {"passthrough": "torch.nn.Dropout", "params": {"p": {"type": _NUM, "default": 0.5}}},
    "dropout2d": {"passthrough": "torch.nn.Dropout2d", "params": {"p": {"type": _NUM, "default": 0.5}}},
    # Shape
    "flatten": {"passthrough": "torch.nn.Flatten", "params": {
        "start_dim": {"type": _INT, "default": 1}, "end_dim": {"type": _INT, "default": -1}}},
    "unflatten": {"params": {
        "dim": {"type": _INT, "default": -1},
        "sizes": {"type": "array", "required": True}}},
    # Embedding / attention / sequence
    "embedding": {"params": {
        "num_embeddings": {"type": _INT, "required": True},
        "embedding_dim": {"type": _INT, "required": True}}},
    "positional_encoding": {"params": {
        "d_model": {"type": _INT, "required": True},
        "max_len": {"type": _INT, "default": 512},
        "learned": {"type": _BOOL, "default": False}}},
    "mha": {"params": {
        "embed_dim": {"type": _INT, "required": True},
        "num_heads": {"type": _INT, "default": 1},
        "dropout": {"type": _NUM, "default": 0.0}}},
    "transformer_encoder": {"params": {
        "d_model": {"type": _INT, "required": True},
        "nhead": {"type": _INT, "default": 1},
        "num_layers": {"type": _INT, "default": 1},
        "dim_feedforward": {"type": _INT, "default": 2048},
        "dropout": {"type": _NUM, "default": 0.0},
        "activation": {"type": _STR, "enum": ["relu", "gelu"], "default": "relu"},
        "causal": {"type": _BOOL, "default": False},
        "norm_first": {"type": _BOOL, "default": False}}},
    "lstm": {"params": {
        "hidden_size": {"type": _INT, "required": True},
        "num_layers": {"type": _INT, "default": 1},
        "dropout": {"type": _NUM, "default": 0.0},
        "input_size": {"type": [_INT, "null"], "default": None}}},
    "gru": {"params": {
        "hidden_size": {"type": _INT, "required": True},
        "num_layers": {"type": _INT, "default": 1},
        "dropout": {"type": _NUM, "default": 0.0},
        "input_size": {"type": [_INT, "null"], "default": None}}},
}


def unknown_layer_params(layer_type: str, params: dict | None) -> list[str]:
    """Return param names a strict layer type would silently ignore.

    Init keys are always accepted. Passthrough layers return ``[]`` because
    their torch constructor validates names itself.

    Args:
        layer_type: ``LAYER_REGISTRY`` key.
        params: The layer's ``params`` mapping from the config.

    Returns:
        Sorted list of rejected names (empty when everything is known).
    """
    spec = LAYER_SPECS.get(layer_type)
    if spec is None or "passthrough" in spec:
        return []
    allowed = set(spec["params"]) | set(INIT_SPEC_KEYS)
    return sorted(k for k in (params or {}) if k not in allowed)
