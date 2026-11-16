"""Output-layer initialization and the continuous policy's scale options.

Covers the fix for heads that build more than one output layer (a
continuous stochastic policy builds one per distribution parameter): one
``output_config`` entry used to reach only the first layer, leaving the
scale layer at PyTorch's default init. Also covers ``initial_scale``,
``state_dependent_scale``, and the legacy ``'kernel params'`` spelling.
"""

from __future__ import annotations

import json
import math

import pytest
import torch as T

from phoenx.env_wrapper import GymnasiumWrapper
from phoenx.models import (
    ModularModel,
    StochasticContinuousHead,
    StochasticContinuousPolicy,
    build_model,
    softplus_inverse,
)

DEVICE = "cpu"
LC = [
    {"type": "dense", "params": {"units": 32, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "elu"},
]
SMALL = {"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}
UNIT = {"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 1.0}}}
OPT = {"type": "Adam", "params": {"lr": 1e-2}}


@pytest.fixture(scope="module")
def pendulum():
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


def _model(env, **head_kwargs) -> ModularModel:
    head_kwargs.setdefault("output_config", [SMALL])
    head = StochasticContinuousHead(
        env, layer_config=list(LC), optimizer_params=OPT, distribution="normal", device=DEVICE, **head_kwargs
    )
    return ModularModel(env=env, branches={"policy": head}, optimizer_params=OPT, device=DEVICE)


def _sigma(dist) -> T.Tensor:
    base = dist
    while hasattr(base, "base_dist"):
        base = base.base_dist
    while hasattr(base, "dist"):
        base = base.dist
    return base.scale


def _features(n: int = 16) -> T.Tensor:
    # No roots or trunk, so the head reads the raw Pendulum observation (3-D).
    return T.randn(n, 3) * 3.0


def test_softplus_inverse():
    for v in (1e-4, 0.05, 0.3, 0.69, 2.0, 30.0):
        assert math.isclose(float(T.nn.functional.softplus(T.tensor(softplus_inverse(v)))), v, rel_tol=1e-5)
    with pytest.raises(ValueError):
        softplus_inverse(0.0)


def test_one_output_config_entry_reaches_both_layers(pendulum):
    """The bug: one entry used to initialize only the mean layer."""
    head = _model(pendulum).branches["policy"]
    for name in ("policy_output_param_1", "policy_output_param_2"):
        layer = head.output_layer[name]
        assert T.count_nonzero(layer.bias) == 0, name  # PhoenX zeroes biases
        # gain 0.01 orthogonal rows have norm 0.01 (PyTorch default would be ~0.5)
        assert float(layer.weight.norm(dim=1).max()) < 0.02, name
    # So the scale starts at softplus(~0) everywhere.
    sigma = _sigma(head(_features()))
    assert float(sigma.min()) > 0.55 and float(sigma.max()) < 0.85


def test_one_entry_per_output_layer_is_paired(pendulum):
    head = _model(pendulum, output_config=[SMALL, UNIT]).branches["policy"]
    mean_norm = float(head.output_layer["policy_output_param_1"].weight.norm(dim=1).max())
    scale_norm = float(head.output_layer["policy_output_param_2"].weight.norm(dim=1).max())
    assert mean_norm < 0.02 and scale_norm > 0.5


def test_wrong_number_of_entries_is_an_error(pendulum):
    with pytest.raises(ValueError, match="output layer"):
        _model(pendulum, output_config=[SMALL, SMALL, SMALL])


def test_legacy_kernel_params_spelling_is_honoured(pendulum):
    legacy = {"type": "dense", "params": {"kernel": "orthogonal", "kernel params": {"gain": 0.01}}}
    head = _model(pendulum, output_config=[legacy]).branches["policy"]
    assert float(head.output_layer["policy_output_param_1"].weight.norm(dim=1).max()) < 0.02


@pytest.mark.parametrize("state_dependent", [True, False])
def test_initial_scale_is_exact(pendulum, state_dependent):
    head = _model(pendulum, initial_scale=0.3, state_dependent_scale=state_dependent).branches["policy"]
    sigma = _sigma(head(_features()))
    assert T.allclose(sigma, T.full_like(sigma, 0.3), atol=1e-5)


def test_state_independent_scale_is_one_learned_value_per_action(pendulum):
    model = _model(pendulum, initial_scale=0.3, state_dependent_scale=False)
    head = model.branches["policy"]
    assert "policy_output_param_2" not in head.output_layer
    assert tuple(head.policy_output_scale.shape) == (head.num_actions,)
    # Same scale in every state.
    s1, s2 = _sigma(head(_features())), _sigma(head(_features()))
    assert T.equal(s1, s2)
    # It is optimized with the head.
    in_optimizer = any(
        p is head.policy_output_scale
        for opt in model.optimizers.values()
        for group in opt.param_groups
        for p in group["params"]
    )
    assert in_optimizer
    before = head.policy_output_scale.detach().clone()
    for opt in model.optimizers.values():
        opt.zero_grad()
    _sigma(head(_features())).sum().backward()
    for opt in model.optimizers.values():
        opt.step()
    assert not T.equal(before, head.policy_output_scale.detach())


@pytest.mark.parametrize("kwargs", [{"initial_scale": 0.3}, {"state_dependent_scale": False}])
def test_scale_options_reject_non_normal(pendulum, kwargs):
    with pytest.raises(ValueError, match="normal"):
        StochasticContinuousHead(pendulum, layer_config=list(LC), distribution="beta", device=DEVICE, **kwargs)


def test_initial_scale_must_be_positive(pendulum):
    with pytest.raises(ValueError, match="> 0"):
        StochasticContinuousHead(pendulum, layer_config=list(LC), distribution="normal", initial_scale=0.0, device=DEVICE)


@pytest.mark.parametrize("state_dependent", [True, False])
def test_config_roundtrip_and_state_load(pendulum, state_dependent):
    model = _model(pendulum, initial_scale=0.3, state_dependent_scale=state_dependent)
    config = model.get_config()
    policy_cfg = config["config"]["branches"]["policy"]["config"]
    assert policy_cfg["initial_scale"] == 0.3
    assert policy_cfg["state_dependent_scale"] is state_dependent
    rebuilt = build_model(json.loads(json.dumps(config)), env=pendulum)
    assert rebuilt.get_config() == config
    assert [n for n, _ in rebuilt.named_parameters()] == [n for n, _ in model.named_parameters()]
    rebuilt.load_state_dict(model.state_dict())
    x = _features()
    assert T.allclose(_sigma(model.branches["policy"](x)), _sigma(rebuilt.branches["policy"](x)))


def test_default_head_still_loads_old_state_dict(pendulum):
    """Defaults keep the parameter layout, so checkpoints saved before the fix load."""
    old = _model(pendulum)
    new = _model(pendulum)
    keys = set(new.state_dict())
    assert any(k.endswith("policy_output_param_2.weight") for k in keys)
    new.load_state_dict(old.state_dict())


@pytest.mark.parametrize("state_dependent", [True, False])
def test_legacy_policy_scale_options(pendulum, state_dependent):
    policy = StochasticContinuousPolicy(
        pendulum,
        layer_config=list(LC),
        output_config=[SMALL],
        distribution="normal",
        initial_scale=0.3,
        state_dependent_scale=state_dependent,
        device=DEVICE,
    )
    sigma = _sigma(policy(T.randn(8, 3)))
    assert T.allclose(sigma, T.full_like(sigma, 0.3), atol=1e-5)
    clone = policy.clone(copy_weights=True)
    assert clone.initial_scale == 0.3 and clone.state_dependent_scale is state_dependent


def test_legacy_policy_one_entry_reaches_both_layers(pendulum):
    policy = StochasticContinuousPolicy(
        pendulum, layer_config=list(LC), output_config=[SMALL], distribution="normal", device=DEVICE
    )
    for name in ("policy_output_param_1", "policy_output_param_2"):
        assert float(policy.output_layer[name].weight.norm(dim=1).max()) < 0.02, name
