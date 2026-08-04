"""Tree utilities for Dict (multi-modal) observations.

Observations flow through the stack either as a single ``torch.Tensor`` (flat
envs) or as a ``dict[str, Tensor]`` (multi-modal envs, one entry per modality).
These helpers let buffers / wrappers / trainers / agents treat both uniformly
without generic pytree recursion in hot loops (observation dicts are shallow:
one level of string keys).

Conventions:
    - "obs" arguments may be a Tensor or a dict[str, Tensor];
    - functions preserve the input's form (tensor in -> tensor out);
    - per-key storage keeps each modality's dtype (uint8 images stay uint8 in
      buffers; models cast/scale at their input boundary).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as T

ObsLike = Union[T.Tensor, Dict[str, T.Tensor]]


def is_obs_dict(obs: Any) -> bool:
    return isinstance(obs, dict)


def tree_map(fn: Callable[[T.Tensor], T.Tensor], obs: ObsLike) -> ObsLike:
    """Apply ``fn`` to every leaf tensor (identity structure)."""
    if isinstance(obs, dict):
        return {k: fn(v) for k, v in obs.items()}
    return fn(obs)


def tree_index(obs: ObsLike, idx) -> ObsLike:
    """``obs[idx]`` per leaf."""
    if isinstance(obs, dict):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


def tree_assign(buf: ObsLike, idx, value: ObsLike) -> None:
    """``buf[idx] = value`` per leaf (dtype-preserving for storage buffers)."""
    if isinstance(buf, dict):
        for k in buf:
            buf[k][idx] = value[k].to(device=buf[k].device, dtype=buf[k].dtype)
    else:
        buf[idx] = value.to(device=buf.device, dtype=buf.dtype)


def tree_cat(items: list, dim: int = 0) -> ObsLike:
    """``torch.cat`` per leaf across a list of same-structured obs."""
    first = items[0]
    if isinstance(first, dict):
        return {k: T.cat([item[k] for item in items], dim=dim) for k in first}
    return T.cat(items, dim=dim)


def tree_stack(items: list, dim: int = 0) -> ObsLike:
    """``torch.stack`` per leaf across a list of same-structured obs."""
    first = items[0]
    if isinstance(first, dict):
        return {k: T.stack([item[k] for item in items], dim=dim) for k in first}
    return T.stack(items, dim=dim)


def tree_clone(obs: ObsLike) -> ObsLike:
    return tree_map(lambda x: x.clone(), obs)


def tree_detach_clone(obs: ObsLike) -> ObsLike:
    return tree_map(lambda x: x.detach().clone(), obs)


def tree_to(obs: ObsLike, device=None, dtype=None) -> ObsLike:
    def _to(x: T.Tensor) -> T.Tensor:
        if dtype is not None:
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)
    return tree_map(_to, obs)


def tree_zero_(obs: ObsLike) -> None:
    if isinstance(obs, dict):
        for v in obs.values():
            v.zero_()
    else:
        obs.zero_()


def flatten_leading(obs: ObsLike, n_dims: int = 2) -> ObsLike:
    """Fold the leading ``n_dims`` dims into one per leaf:
    ``(A, B, *feat) -> (A*B, *feat)`` (feature shape preserved — image leaves
    are NOT flattened, unlike a bare ``reshape(total, -1)``).
    """
    def _flat(x: T.Tensor) -> T.Tensor:
        lead = int(np.prod(x.shape[:n_dims]))
        return x.reshape(lead, *x.shape[n_dims:])
    return tree_map(_flat, obs)


def unflatten_leading(obs: ObsLike, dims: Tuple[int, ...]) -> ObsLike:
    """Inverse of :func:`flatten_leading`: ``(A*B, *feat) -> (*dims, *feat)``."""
    return tree_map(lambda x: x.reshape(*dims, *x.shape[1:]), obs)


def obs_batch_size(obs: ObsLike) -> int:
    if isinstance(obs, dict):
        return next(iter(obs.values())).shape[0]
    return obs.shape[0]


# -----------------------------------------------------------------------------
# Space-driven allocation / spec extraction (used by buffers and wrappers)
# -----------------------------------------------------------------------------

_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.float32): T.float32,
    np.dtype(np.float64): T.float32,   # store float64 obs as float32 (legacy behavior)
    np.dtype(np.uint8): T.uint8,       # images stay uint8 in storage
    np.dtype(np.int64): T.long,
    np.dtype(np.int32): T.long,
    np.dtype(np.bool_): T.bool,
}


def torch_dtype_for(space: gym.Space) -> T.dtype:
    return _NUMPY_TO_TORCH_DTYPE.get(np.dtype(space.dtype), T.float32)


def obs_spec_from_space(space: gym.Space, obs_key: str | None,
                        goal_keys: Tuple[str | None, ...] = ()) -> Union[Tuple, Dict[str, Tuple]]:
    """Extract the per-key storage spec for observations.

    Returns either a single ``(shape, dtype)`` tuple (flat obs or Dict space
    reduced by ``obs_key``) or ``{key: (shape, dtype)}`` for multi-modal Dict
    spaces (``obs_key is None``), excluding goal keys.

    Flat (single-tensor) observations keep the legacy float32 storage dtype;
    per-key dtypes (uint8 images, etc.) are only honored for multi-modal Dict
    storage where the memory savings matter and models cast at their boundary.
    """
    if isinstance(space, gym.spaces.Dict):
        if obs_key is not None:
            sub = space.spaces[obs_key]
            return (tuple(sub.shape), T.float32)
        excluded = {k for k in goal_keys if k is not None}
        return {
            k: (tuple(sub.shape), torch_dtype_for(sub))
            for k, sub in space.spaces.items() if k not in excluded
        }
    return (tuple(space.shape), T.float32)


def alloc_from_spec(spec, prefix_shape: Tuple[int, ...], device) -> ObsLike:
    """Preallocate zero storage with the given leading dims per spec leaf."""
    if isinstance(spec, dict):
        return {
            k: T.zeros((*prefix_shape, *shape), dtype=dtype, device=device)
            for k, (shape, dtype) in spec.items()
        }
    shape, dtype = spec
    return T.zeros((*prefix_shape, *shape), dtype=dtype, device=device)


def flatten_obs(obs: ObsLike, keys=None) -> T.Tensor:
    """Flatten (and for dicts, concatenate) an observation into ``(B, D)``.

    Used by consumers that need a single flat vector view (e.g. intrinsic
    motivation modules). ``keys`` selects/orders dict entries (default:
    declaration order). uint8 leaves are scaled to [0, 1] floats.
    """
    def _flat(x: T.Tensor) -> T.Tensor:
        if x.dtype == T.uint8:
            x = x.float() / 255.0
        elif x.dtype != T.float32:
            x = x.float()
        return x.reshape(x.shape[0], -1)

    if isinstance(obs, dict):
        selected = keys if keys is not None else list(obs.keys())
        parts = [_flat(obs[k]) for k in selected]
        return parts[0] if len(parts) == 1 else T.cat(parts, dim=-1)
    return _flat(obs)
