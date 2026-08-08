"""Unit tests for ``src/phoenx/intrinsic_motivation.py``.

These tests import the real classes from the PhoenX API and exercise them
directly so that any refactor of the API is automatically validated by simply
re-running the test file (no test-side mock copies of the classes).

Covers:
    * ICM, RND, EpisodicNovelty, CompositeIntrinsicMotivation
    * additive / multiplicative / max / ngu combination rules
    * The intrinsic-motivation registry and ``IntrinsicMotivation.load``
      dispatcher
    * ``EpisodicNovelty.on_episode_end`` memory clearing semantics
    * Save / load round-trip for every concrete subclass

The tests use small CPU networks and minimal vec environments
(``CartPole-v1`` for discrete actions, ``Pendulum-v1`` for continuous
actions) so the full suite runs in seconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch as T

from phoenx.env_wrapper import GymnasiumWrapper
from phoenx.intrinsic_motivation import (
    ICM,
    RND,
    CompositeIntrinsicMotivation,
    EpisodicNovelty,
    IntrinsicMotivation,
    _COMBINATION_RULES,
    _REGISTRY,
    additive_combination,
    max_combination,
    multiplicative_combination,
    ngu_combination,
    register_intrinsic_motivation,
)
from phoenx.normalizer import RewardNorm
from phoenx.schedulers import ScheduleWrapper

DEVICE = "cpu"
NUM_ENVS = 2

# ----------------------------------------------------------------------------- 
# Determinism
# -----------------------------------------------------------------------------
T.manual_seed(0)
np.random.seed(0)


# -----------------------------------------------------------------------------
# Real-environment fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def discrete_env() -> GymnasiumWrapper:
    """Vector CartPole-v1 (Discrete(2) actions, Box(4,) obs)."""
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=NUM_ENVS, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def continuous_env() -> GymnasiumWrapper:
    """Vector Pendulum-v1 (Box(1,) actions, Box(3,) obs)."""
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=NUM_ENVS, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _obs_dim(env: GymnasiumWrapper) -> tuple[int, ...]:
    return env.single_observation_space.shape


def _action_dim(env: GymnasiumWrapper) -> tuple[int, ...]:
    space = env.single_action_space
    if hasattr(space, "n"):
        return (int(space.n),)
    return space.shape


def _is_discrete(env: GymnasiumWrapper) -> bool:
    return hasattr(env.single_action_space, "n")


def _state_dicts_match(ref_sd: dict, loaded_sd: dict) -> None:
    """Compare two state_dicts even when they live on different devices.

    The intrinsic-motivation ``_load_impl`` methods do not persist the original
    ``device`` choice in their ``get_config()``, so the loaded model may sit on
    a different device than the original. We compare values after moving both
    sides to CPU.
    """
    assert ref_sd.keys() == loaded_sd.keys()
    for k in ref_sd:
        ref = ref_sd[k]
        loaded = loaded_sd[k]
        if isinstance(ref, T.Tensor) and ref.dtype.is_floating_point:
            assert T.allclose(ref.cpu(), loaded.cpu()), f"{k} differs after load"


def _to_cpu(model):
    """Move an intrinsic-motivation model to CPU for portable comparisons."""
    model.to("cpu")
    if hasattr(model, "device"):
        model.device = T.device("cpu")
    return model


def _random_batch(env: GymnasiumWrapper, batch: int | None = None) -> dict[str, T.Tensor]:
    """Construct dummy ``states / next_states / actions`` for an env."""
    if batch is None:
        batch = NUM_ENVS
    obs_shape = _obs_dim(env)
    states = T.randn((batch, *obs_shape), dtype=T.float32, device=DEVICE)
    next_states = T.randn((batch, *obs_shape), dtype=T.float32, device=DEVICE)
    if _is_discrete(env):
        n = int(env.single_action_space.n)
        actions = T.randint(0, n, (batch,), dtype=T.long, device=DEVICE)
    else:
        a_shape = env.single_action_space.shape
        actions = T.randn((batch, *a_shape), dtype=T.float32, device=DEVICE)
    return {"states": states, "next_states": next_states, "actions": actions}


# -----------------------------------------------------------------------------
# Model-config builders (kept small for speed)
# -----------------------------------------------------------------------------
def icm_configs(with_encoder: bool = False) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if with_encoder:
        cfg["encoder"] = {
            "layer_config": [
                {"type": "dense", "params": {"units": 16}},
                {"type": "relu"},
            ],
            "output_layer": [{"type": "dense", "params": {"units": 8}}],
        }
    cfg["inverse_model"] = {
        "layer_config": [
            {"type": "dense", "params": {"units": 16}},
            {"type": "relu"},
        ],
        "output_layer": [{"type": "dense", "params": {}}],
    }
    cfg["forward_model"] = {
        "layer_config": [
            {"type": "dense", "params": {"units": 16}},
            {"type": "relu"},
        ],
        "output_layer": [{"type": "dense", "params": {}}],
    }
    return cfg


def rnd_configs(output_dim: int = 16) -> dict[str, Any]:
    base = {
        "layer_config": [
            {"type": "dense", "params": {"units": 16}},
            {"type": "relu"},
        ],
        "output_layer": [{"type": "dense", "params": {"units": output_dim}}],
    }
    return {"target": base, "predictor": base}


def episodic_configs(embed_dim: int = 8) -> dict[str, Any]:
    return {
        "encoder": {
            "layer_config": [
                {"type": "dense", "params": {"units": 16}},
                {"type": "relu"},
            ],
            "output_layer": [{"type": "dense", "params": {"units": embed_dim}}],
        },
        "inverse_model": {
            "layer_config": [
                {"type": "dense", "params": {"units": 16}},
                {"type": "relu"},
            ],
            "output_layer": [{"type": "dense", "params": {}}],
        },
    }


OPT_PARAMS = {"type": "Adam", "params": {"lr": 1e-3}}


# =============================================================================
# Registry tests
# =============================================================================
class TestRegistry:
    def test_builtin_subclasses_registered(self):
        for name in ("ICM", "RND", "EpisodicNovelty", "CompositeIntrinsicMotivation"):
            assert name in _REGISTRY, f"{name} missing from intrinsic-motivation registry"
            assert issubclass(_REGISTRY[name], IntrinsicMotivation)

    def test_register_decorator_adds_class(self):
        @register_intrinsic_motivation
        class _Dummy(IntrinsicMotivation):
            def train(self, states, next_states, actions=None):
                return T.tensor(0.0)

            def get_config(self):
                return {}

        try:
            assert "_Dummy" in _REGISTRY
            assert _REGISTRY["_Dummy"] is _Dummy
        finally:
            _REGISTRY.pop("_Dummy", None)

    def test_create_instance_unknown_raises(self, discrete_env):
        with pytest.raises(ValueError, match="Unknown intrinsic motivation type"):
            IntrinsicMotivation.create_instance("DoesNotExist", env=discrete_env)


# =============================================================================
# Combination-function tests
# =============================================================================
class TestCombinationFunctions:
    def test_additive_default_weights(self):
        a = T.tensor([1.0, 2.0])
        b = T.tensor([3.0, 4.0])
        out = additive_combination([a, b])
        assert T.allclose(out, T.tensor([4.0, 6.0]))

    def test_additive_custom_weights(self):
        a = T.tensor([1.0, 2.0])
        b = T.tensor([3.0, 4.0])
        out = additive_combination([a, b], weights=[0.5, 2.0])
        assert T.allclose(out, T.tensor([0.5 * 1 + 2.0 * 3, 0.5 * 2 + 2.0 * 4]))

    def test_multiplicative(self):
        a = T.tensor([1.0, 2.0])
        b = T.tensor([3.0, 4.0])
        c = T.tensor([2.0, 0.5])
        out = multiplicative_combination([a, b, c])
        assert T.allclose(out, T.tensor([1 * 3 * 2, 2 * 4 * 0.5]))

    def test_max(self):
        a = T.tensor([1.0, 5.0, 2.0])
        b = T.tensor([4.0, 1.0, 3.0])
        out = max_combination([a, b])
        assert T.allclose(out, T.tensor([4.0, 5.0, 3.0]))

    def test_ngu_combination_clamps_lifelong(self):
        epi = T.tensor([2.0, 2.0, 2.0])
        lifelong = T.tensor([0.5, 3.0, 10.0])  # clamped to [1, L]
        out = ngu_combination([epi, lifelong], L=5.0)
        expected = T.tensor([2.0 * 1.0, 2.0 * 3.0, 2.0 * 5.0])
        assert T.allclose(out, expected)

    def test_ngu_combination_arg_count(self):
        a = T.tensor([1.0])
        with pytest.raises(ValueError, match="exactly 2 rewards"):
            ngu_combination([a])
        with pytest.raises(ValueError, match="exactly 2 rewards"):
            ngu_combination([a, a, a])

    def test_additive_empty_raises(self):
        with pytest.raises(ValueError, match="Empty rewards list"):
            additive_combination([])

    def test_multiplicative_empty_raises(self):
        with pytest.raises(ValueError, match="Empty rewards list"):
            multiplicative_combination([])

    def test_max_empty_raises(self):
        with pytest.raises(ValueError, match="Empty rewards list"):
            max_combination([])

    def test_combination_rule_registry(self):
        assert set(_COMBINATION_RULES.keys()) == {
            "additive",
            "multiplicative",
            "max",
            "ngu",
        }
        assert _COMBINATION_RULES["additive"] is additive_combination
        assert _COMBINATION_RULES["multiplicative"] is multiplicative_combination
        assert _COMBINATION_RULES["max"] is max_combination
        assert _COMBINATION_RULES["ngu"] is ngu_combination


# =============================================================================
# ICM tests
# =============================================================================
class TestICM:
    @pytest.fixture
    def icm_discrete(self, discrete_env):
        return ICM(
            env=discrete_env,
            model_configs=icm_configs(with_encoder=False),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )

    @pytest.fixture
    def icm_continuous(self, continuous_env):
        return ICM(
            env=continuous_env,
            model_configs=icm_configs(with_encoder=False),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )

    @pytest.fixture
    def icm_discrete_with_encoder(self, discrete_env):
        return ICM(
            env=discrete_env,
            model_configs=icm_configs(with_encoder=True),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )

    @pytest.fixture
    def icm_continuous_with_encoder(self, continuous_env):
        return ICM(
            env=continuous_env,
            model_configs=icm_configs(with_encoder=True),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )

    def test_init_discrete_no_encoder(self, icm_discrete, discrete_env):
        assert icm_discrete.is_online is False
        assert icm_discrete._is_discrete is True
        assert icm_discrete._use_encoder is False
        assert icm_discrete.encoder is None
        assert icm_discrete.inverse_model is not None
        assert icm_discrete.forward_model is not None
        assert icm_discrete.action_dim == _action_dim(discrete_env)
        assert icm_discrete.obs_dim == _obs_dim(discrete_env)
        assert icm_discrete.optimizer is not None
        # Inverse output size should equal num discrete actions
        last = list(icm_discrete.inverse_model.values())[-1]
        assert last.out_features == int(np.prod(_action_dim(discrete_env)))
        # Forward output size should equal obs dim (no encoder)
        last_fwd = list(icm_discrete.forward_model.values())[-1]
        assert last_fwd.out_features == int(np.prod(_obs_dim(discrete_env)))

    def test_init_continuous_no_encoder(self, icm_continuous, continuous_env):
        assert icm_continuous.is_online is False
        assert icm_continuous._is_discrete is False
        assert icm_continuous._use_encoder is False
        assert icm_continuous.action_dim == _action_dim(continuous_env)
        assert icm_continuous.obs_dim == _obs_dim(continuous_env)

    def test_init_discrete_with_encoder(self, icm_discrete_with_encoder):
        assert icm_discrete_with_encoder.is_online is False
        assert icm_discrete_with_encoder._use_encoder is True
        assert icm_discrete_with_encoder.encoder is not None
        # Forward output should match encoder output (8)
        last_fwd = list(icm_discrete_with_encoder.forward_model.values())[-1]
        assert last_fwd.out_features == 8

    def test_init_continuous_with_encoder(self, icm_continuous_with_encoder):
        assert icm_continuous_with_encoder.is_online is False
        assert icm_continuous_with_encoder._use_encoder is True
        assert icm_continuous_with_encoder.encoder is not None
        # Forward output should match encoder output (8)
        last_fwd = list(icm_continuous_with_encoder.forward_model.values())[-1]
        assert last_fwd.out_features == 8

    def test_compute_learn_reward_shape_and_sign(self, icm_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=4)
        r = icm_discrete.compute_learn_reward(**batch)
        assert isinstance(r, T.Tensor)
        assert r.shape == (4,)
        assert r.device.type == DEVICE
        # Squared-error * non-negative scale must be >= 0
        assert (r >= 0).all().item()
        # No gradients leak from inference helper
        assert r.requires_grad is False

    def test_compute_learn_reward_continuous(self, icm_continuous, continuous_env):
        batch = _random_batch(continuous_env, batch=4)
        r = icm_continuous.compute_learn_reward(**batch)
        assert r.shape == (4,)
        assert (r >= 0).all().item()

    def test_train_step_reduces_or_updates_params(self, icm_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=8)
        # Snapshot a parameter
        params_before = [p.detach().clone() for p in icm_discrete.parameters()]
        loss = icm_discrete.train(**batch)
        assert isinstance(loss, T.Tensor)
        assert loss.ndim == 0
        assert T.isfinite(loss).item()
        # At least one parameter should have changed
        params_after = list(icm_discrete.parameters())
        any_changed = any(
            not T.equal(a, b) for a, b in zip(params_before, params_after)
        )
        assert any_changed, "ICM.train did not update any parameters"

    def test_train_step_continuous(self, icm_continuous, continuous_env):
        batch = _random_batch(continuous_env, batch=8)
        loss = icm_continuous.train(**batch)
        assert T.isfinite(loss).item()

    # ----- Math correctness ------------------------------------------------
    def test_compute_learn_reward_matches_documented_formula(
        self, icm_discrete, discrete_env
    ):
        """Public reward = 0.5 * scaled_weight * ||f(φ(s),a) - φ(s')||².

        We re-derive the expected error using the model's own internal
        ``_full_forward`` (the same computation the public API uses), then
        verify the public output matches bit-for-bit up to FP noise. This
        validates the formula AND the wiring without being brittle to layer
        topology changes.
        """
        icm = icm_discrete
        batch = _random_batch(discrete_env, batch=8)
        with T.no_grad():
            _, pred_ns, encoded_ns = icm._full_forward(**batch)
            err = (pred_ns - encoded_ns).pow(2).sum(dim=-1)
            expected = 0.5 * icm._scaled_reward_weight() * err
        actual = icm.compute_learn_reward(**batch)
        assert T.allclose(actual, expected, atol=1e-6)

    def test_compute_learn_reward_matches_formula_continuous(
        self, icm_continuous, continuous_env
    ):
        icm = icm_continuous
        batch = _random_batch(continuous_env, batch=8)
        with T.no_grad():
            _, pred_ns, encoded_ns = icm._full_forward(**batch)
            err = (pred_ns - encoded_ns).pow(2).sum(dim=-1)
            expected = 0.5 * icm._scaled_reward_weight() * err
        actual = icm.compute_learn_reward(**batch)
        assert T.allclose(actual, expected, atol=1e-6)

    def test_compute_learn_reward_scales_linearly_with_reward_weight(
        self, discrete_env
    ):
        """Doubling ``reward_weight`` should exactly double the reward."""
        cfg = icm_configs()
        icm_a = ICM(
            env=discrete_env,
            model_configs=cfg,
            optimizer_params=OPT_PARAMS,
            reward_weight=1.0,
            device=DEVICE,
        )
        icm_b = ICM(
            env=discrete_env,
            model_configs=cfg,
            optimizer_params=OPT_PARAMS,
            reward_weight=2.0,
            device=DEVICE,
        )
        # Identical weights ⇒ identical raw error.
        icm_b.load_state_dict(icm_a.state_dict())
        batch = _random_batch(discrete_env, batch=8)
        r_a = icm_a.compute_learn_reward(**batch)
        r_b = icm_b.compute_learn_reward(**batch)
        assert T.allclose(r_b, 2.0 * r_a, atol=1e-6)

    def test_train_loss_matches_beta_weighted_formula(self, discrete_env):
        """Loss = (1 - β) * L_inverse + β * 0.5 * MSE(pred_ns, encoded_ns)."""
        beta = 0.3
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            beta=beta,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=16)
        # The networks have no stateful layers (dense + relu only), so the
        # forward result is identical between eval and train modes.
        with T.no_grad():
            pred_a, pred_ns, encoded_ns = icm._full_forward(**batch)
            if icm._is_discrete:
                inv = T.nn.CrossEntropyLoss()(pred_a, batch["actions"].long().view(-1))
            else:
                inv = T.nn.MSELoss()(pred_a, batch["actions"].float())
            fwd = 0.5 * T.nn.MSELoss()(pred_ns, encoded_ns)
            expected = (1 - beta) * inv + beta * fwd
        # train() returns the loss before stepping the optimizer.
        loss = icm.train(**batch)
        assert T.allclose(loss.detach(), expected, atol=1e-5)

    def test_train_loss_matches_formula_continuous(self, continuous_env):
        beta = 0.4
        icm = ICM(
            env=continuous_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            beta=beta,
            device=DEVICE,
        )
        batch = _random_batch(continuous_env, batch=16)
        with T.no_grad():
            pred_a, pred_ns, encoded_ns = icm._full_forward(**batch)
            inv = T.nn.MSELoss()(pred_a, batch["actions"].float())
            fwd = 0.5 * T.nn.MSELoss()(pred_ns, encoded_ns)
            expected = (1 - beta) * inv + beta * fwd
        loss = icm.train(**batch)
        assert T.allclose(loss.detach(), expected, atol=1e-5)

    def test_compute_learn_reward_feeds_reward_normalizer(self, discrete_env):
        """The reward normalizer's ``step`` counter advances on each call."""
        norm = RewardNorm(gamma=0.99, clip_value=5.0, device=DEVICE)
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            reward_normalizer=norm,
            device=DEVICE,
        )
        assert norm.step == 0
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        _ = icm.compute_learn_reward(**batch)
        assert norm.step == 1
        _ = icm.compute_learn_reward(**batch)
        assert norm.step == 2

    def test_get_config_keys(self, icm_discrete):
        cfg = icm_discrete.get_config()
        for key in (
            "env",
            "model_configs",
            "optimizer_params",
            "reward_weight",
            "reward_scheduler",
            "beta",
            "extrinsic_threshold",
            "reward_normalizer",
        ):
            assert key in cfg
        assert cfg["beta"] == 0.2  # default
        assert cfg["reward_scheduler"] is None
        assert cfg["reward_normalizer"] is None

    def test_reward_weight_and_scheduler_scale(self, discrete_env):
        sched = ScheduleWrapper(
            schedule_type="linear",
            steps=100,
            start_value=1.0,
            end_value=0.0,
        )
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            reward_weight=2.0,
            reward_scheduler=sched,
            device=DEVICE,
        )
        # Internal scaled weight must equal weight * scheduler factor
        scale = icm._scaled_reward_weight()
        assert pytest.approx(scale) == 2.0 * sched.get_factor()

    def test_use_extrinsic_reward_threshold(self, discrete_env):
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            extrinsic_threshold=50,
            device=DEVICE,
        )
        assert icm.use_extrinsic_reward(49) is False
        assert icm.use_extrinsic_reward(50) is True
        assert icm.use_extrinsic_reward(1000) is True

    def test_save_load_roundtrip(self, discrete_env, tmp_path):
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(with_encoder=False),
            optimizer_params=OPT_PARAMS,
            reward_weight=0.3,
            beta=0.4,
            extrinsic_threshold=10,
            device=DEVICE,
        )
        # Run a forward to make sure LazyLinears are materialized before save
        batch = _random_batch(discrete_env, batch=4)
        _ = icm.compute_learn_reward(**batch)
        icm.save(tmp_path)

        # Files exist
        assert (tmp_path / "intrinsic_motivation" / "config.json").exists()
        assert (tmp_path / "intrinsic_motivation" / "pytorch_model.pt").exists()

        # Dispatch via base classmethod
        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)
        assert isinstance(loaded, ICM)
        assert loaded.reward_weight == 0.3
        assert loaded.beta == 0.4
        assert loaded.extrinsic_threshold == 10

        _to_cpu(loaded)
        _state_dicts_match(icm.state_dict(), loaded.state_dict())

        with T.no_grad():
            r_ref = icm.compute_learn_reward(**batch)
            r_loaded = loaded.compute_learn_reward(**batch)
        assert T.allclose(r_ref.cpu(), r_loaded.cpu(), atol=1e-5)


# =============================================================================
# RND tests
# =============================================================================
class TestRND:
    @pytest.fixture
    def rnd(self, discrete_env):
        return RND(
            env=discrete_env,
            model_configs=rnd_configs(output_dim=16),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )

    def test_init(self, rnd):
        assert rnd.is_online is False
        assert rnd.target is not None
        assert rnd.predictor is not None
        # Target frozen
        assert all(not p.requires_grad for p in rnd.target.parameters())
        # Predictor trainable
        assert all(p.requires_grad for p in rnd.predictor.parameters())
        # Output dims match
        last_t = list(rnd.target.values())[-1]
        last_p = list(rnd.predictor.values())[-1]
        assert last_t.out_features == 16
        assert last_p.out_features == 16

    def test_optimizer_only_covers_predictor(self, rnd):
        opt_params = {id(p) for group in rnd.optimizer.param_groups for p in group["params"]}
        pred_params = {id(p) for p in rnd.predictor.parameters()}
        target_params = {id(p) for p in rnd.target.parameters()}
        assert opt_params == pred_params
        assert opt_params.isdisjoint(target_params)

    def test_default_output_dim_constant(self):
        assert RND.DEFAULT_OUTPUT_DIM == 512

    def test_compute_learn_reward_shape(self, rnd, discrete_env):
        batch = _random_batch(discrete_env, batch=4)
        r = rnd.compute_learn_reward(**batch)
        assert r.shape == (4,)
        assert (r >= 0).all().item()

    def test_train_updates_predictor_only(self, rnd, discrete_env):
        # Snapshot before
        target_before = [p.detach().clone() for p in rnd.target.parameters()]
        pred_before = [p.detach().clone() for p in rnd.predictor.parameters()]
        batch = _random_batch(discrete_env, batch=8)
        loss = rnd.train(**batch)
        assert T.isfinite(loss).item()
        # Target untouched
        for a, b in zip(target_before, rnd.target.parameters()):
            assert T.equal(a, b), "Target parameters changed during RND.train"
        # Predictor changed
        any_changed = any(
            not T.equal(a, b) for a, b in zip(pred_before, rnd.predictor.parameters())
        )
        assert any_changed, "RND.train did not change predictor parameters"

    def test_train_loss_matches_mean_squared_error(self, rnd, discrete_env):
        # Loss is exactly the mean over the batch of the per-sample sum of
        # squared (predictor - target) embedding differences.
        rnd.predictor.eval()
        rnd.target.eval()
        batch = _random_batch(discrete_env, batch=8)
        with T.no_grad():
            t_out, p_out = rnd._embed(batch["next_states"])
            expected = (p_out - t_out).pow(2).sum(dim=-1).mean()
        loss = rnd.train(**batch)
        # Loss is computed *before* the optimizer.step, so it should match
        # `expected` from the same evaluation up to floating-point noise.
        assert T.allclose(loss.detach(), expected, atol=1e-5)

    # ----- Math correctness ------------------------------------------------
    def test_compute_learn_reward_matches_squared_difference(self, rnd, discrete_env):
        """r_i(s') = scaled_weight * ||predictor(s') - target(s')||²."""
        rnd.target.eval()
        rnd.predictor.eval()
        batch = _random_batch(discrete_env, batch=8)
        with T.no_grad():
            t_out, p_out = rnd._embed(batch["next_states"])
            err = (p_out - t_out).pow(2).sum(dim=-1)
            expected = rnd._scaled_reward_weight() * err
        actual = rnd.compute_learn_reward(**batch)
        assert T.allclose(actual, expected, atol=1e-6)

    def test_compute_learn_reward_uses_next_states_not_states(
        self, rnd, discrete_env
    ):
        """RND novelty is over s' only — swapping s for s' must not change r."""
        batch = _random_batch(discrete_env, batch=8)
        ref = rnd.compute_learn_reward(**batch)
        # Replace `states` with garbage; `next_states` is untouched.
        garbage = T.randn_like(batch["states"]) * 1e3
        other = rnd.compute_learn_reward(
            states=garbage,
            next_states=batch["next_states"],
            actions=batch["actions"],
        )
        assert T.allclose(ref, other, atol=1e-6)

    def test_compute_learn_reward_scales_linearly_with_reward_weight(
        self, discrete_env
    ):
        cfg = rnd_configs(output_dim=16)
        a = RND(
            env=discrete_env,
            model_configs=cfg,
            optimizer_params=OPT_PARAMS,
            reward_weight=1.0,
            device=DEVICE,
        )
        b = RND(
            env=discrete_env,
            model_configs=cfg,
            optimizer_params=OPT_PARAMS,
            reward_weight=3.0,
            device=DEVICE,
        )
        b.load_state_dict(a.state_dict())
        batch = _random_batch(discrete_env, batch=8)
        r_a = a.compute_learn_reward(**batch)
        r_b = b.compute_learn_reward(**batch)
        assert T.allclose(r_b, 3.0 * r_a, atol=1e-6)

    def test_compute_learn_reward_feeds_reward_normalizer(self, discrete_env):
        norm = RewardNorm(gamma=0.99, clip_value=5.0, device=DEVICE)
        rnd = RND(
            env=discrete_env,
            model_configs=rnd_configs(output_dim=16),
            optimizer_params=OPT_PARAMS,
            reward_normalizer=norm,
            device=DEVICE,
        )
        assert norm.step == 0
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        _ = rnd.compute_learn_reward(**batch)
        assert norm.step == 1

    def test_get_config_keys(self, rnd):
        cfg = rnd.get_config()
        for key in (
            "env",
            "model_configs",
            "optimizer_params",
            "reward_weight",
            "reward_scheduler",
            "extrinsic_threshold",
            "reward_normalizer",
        ):
            assert key in cfg

    def test_save_load_roundtrip(self, discrete_env, tmp_path):
        rnd = RND(
            env=discrete_env,
            model_configs=rnd_configs(output_dim=16),
            optimizer_params=OPT_PARAMS,
            reward_weight=0.5,
            extrinsic_threshold=5,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=4)
        _ = rnd.compute_learn_reward(**batch)
        rnd.save(tmp_path)

        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)
        assert isinstance(loaded, RND)
        assert loaded.reward_weight == 0.5
        assert loaded.extrinsic_threshold == 5

        # Frozen target stays frozen after load
        assert all(not p.requires_grad for p in loaded.target.parameters())

        _to_cpu(loaded)
        _state_dicts_match(rnd.state_dict(), loaded.state_dict())

        with T.no_grad():
            r_ref = rnd.compute_learn_reward(**batch)
            r_loaded = loaded.compute_learn_reward(**batch)
        assert T.allclose(r_ref.cpu(), r_loaded.cpu(), atol=1e-5)


# =============================================================================
# EpisodicNovelty tests
# =============================================================================
class TestEpisodicNovelty:
    @pytest.fixture
    def epi_discrete(self, discrete_env):
        return EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=8),
            optimizer_params=OPT_PARAMS,
            memory_size=64,
            k=3,
            device=DEVICE,
        )

    @pytest.fixture
    def epi_continuous(self, continuous_env):
        return EpisodicNovelty(
            env=continuous_env,
            model_configs=episodic_configs(embed_dim=8),
            optimizer_params=OPT_PARAMS,
            memory_size=64,
            k=3,
            device=DEVICE,
        )

    def test_init(self, epi_discrete):
        assert epi_discrete.is_online is True
        assert epi_discrete.encoder is not None
        assert epi_discrete.inverse_model is not None
        assert epi_discrete.embed_dim == 8
        assert epi_discrete.num_envs == NUM_ENVS
        assert len(epi_discrete._memories) == NUM_ENVS
        for mem in epi_discrete._memories:
            assert mem.shape == (0, 8)

    def test_memory_grows_with_rollout_reward(self, epi_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        _ = epi_discrete.compute_rollout_reward(
            **batch, env_indices=env_indices
        )
        for mem in epi_discrete._memories:
            assert mem.shape == (1, 8)

        _ = epi_discrete.compute_rollout_reward(
            **batch, env_indices=env_indices
        )
        for mem in epi_discrete._memories:
            assert mem.shape == (2, 8)

    def test_compute_rollout_reward_shape(self, epi_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        r = epi_discrete.compute_rollout_reward(
            **batch, env_indices=env_indices
        )
        assert r.shape == (NUM_ENVS,)
        # On cold memory the bonus is set to 1.0 then scaled by reward_weight
        # (default 1.0) ⇒ output equals 1.0 exactly.
        assert T.allclose(r, T.ones_like(r))

    def test_compute_rollout_reward_default_env_indices(self, epi_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        r = epi_discrete.compute_rollout_reward(**batch)
        assert r.shape == (NUM_ENVS,)

    def test_compute_learn_reward_zero_default(self, epi_discrete, discrete_env):
        """EpisodicNovelty has no learn reward — inherits the zero default."""
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        r = epi_discrete.compute_learn_reward(**batch)
        assert r.shape == (NUM_ENVS,)
        assert T.allclose(r, T.zeros_like(r))

    def test_on_episode_end_clears_memory(self, epi_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        # Add a few items first
        for _ in range(3):
            _ = epi_discrete.compute_rollout_reward(
                **batch, env_indices=env_indices
            )
        for mem in epi_discrete._memories:
            assert mem.shape[0] == 3
        # Clear env 0 only
        epi_discrete.on_episode_end(T.tensor([0], dtype=T.long))
        assert epi_discrete._memories[0].shape == (0, 8)
        assert epi_discrete._memories[1].shape[0] == 3

    def test_memory_size_cap(self, discrete_env):
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=8),
            optimizer_params=OPT_PARAMS,
            memory_size=4,
            k=2,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        for _ in range(10):
            epi.compute_rollout_reward(**batch, env_indices=env_indices)
        for mem in epi._memories:
            assert mem.shape[0] == 4, "Memory exceeded memory_size cap"

    def test_train_step(self, epi_discrete, discrete_env):
        batch = _random_batch(discrete_env, batch=8)
        params_before = [p.detach().clone() for p in epi_discrete.parameters()]
        loss = epi_discrete.train(**batch)
        assert T.isfinite(loss).item()
        any_changed = any(
            not T.equal(a, b)
            for a, b in zip(params_before, epi_discrete.parameters())
        )
        assert any_changed

    def test_train_continuous(self, epi_continuous, continuous_env):
        batch = _random_batch(continuous_env, batch=8)
        loss = epi_continuous.train(**batch)
        assert T.isfinite(loss).item()

    # ----- Math correctness for the kNN bonus ------------------------------
    # Formula (Badia et al., 2020):
    #     α_epi(x) = 1 / sqrt( Σ_{f ∈ N_k(x)} K(x, f) + c )
    #     K(x, f)  = ε / ( d²(x, f) / d̄² + ε )
    # with d̄² updated via EWMA over the top-k squared distances.

    def test_knn_bonus_returns_neutral_for_sparse_memory(self, discrete_env):
        """Memories with < 2 entries → bonus = 1.0 (paper-specified fallback)."""
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=2),
            optimizer_params=OPT_PARAMS,
            memory_size=10,
            k=2,
            device=DEVICE,
        )
        x = T.zeros((1, 2), dtype=T.float32, device=DEVICE)
        # Empty memory
        bonus = epi._knn_bonus(x, T.tensor([0], dtype=T.long, device=DEVICE))
        assert T.allclose(bonus, T.ones_like(bonus))
        # One entry — still < 2
        epi._memories[0] = T.tensor([[1.0, 2.0]], dtype=T.float32)
        bonus = epi._knn_bonus(x, T.tensor([0], dtype=T.long, device=DEVICE))
        assert T.allclose(bonus, T.ones_like(bonus))

    def test_knn_bonus_exact_kernel_value_with_known_memory(self, discrete_env):
        """Single-neighbour query at d² = 0 ⇒ K = ε/ε = 1 ⇒ bonus = 1/√(1 + c)."""
        kernel_eps = 1e-3
        cluster = 8e-3
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=2),
            optimizer_params=OPT_PARAMS,
            memory_size=10,
            k=1,
            kernel_epsilon=kernel_eps,
            cluster_distance=cluster,
            max_similarity=8.0,
            running_mean_decay=0.99,
            device=DEVICE,
        )
        # Hand-injected memory; query equals first memory entry exactly.
        q = T.tensor([0.5, 0.5], dtype=T.float32, device=DEVICE)
        far = T.tensor([10.0, 10.0], dtype=T.float32, device=DEVICE)
        epi._memories[0] = T.stack([q.cpu(), far.cpu()])
        epi._running_sq_dist = T.tensor(1.0, device=DEVICE)

        bonus = epi._knn_bonus(q.unsqueeze(0), T.tensor([0], dtype=T.long, device=DEVICE))

        # d² (top-1, smallest) = 0
        # EWMA: 0.99 * 1.0 + 0.01 * 0 = 0.99
        # d²_n = 0 / (0.99 + 1e-8) = 0
        # K = ε / (0 + ε) = 1
        # sim = 1.0; not capped because max_similarity² = 64 > 1
        # bonus = 1 / sqrt(1 + cluster) = 1 / sqrt(1.008)
        expected = 1.0 / T.sqrt(T.tensor(1.0 + cluster))
        assert T.allclose(bonus.cpu(), expected.unsqueeze(0), atol=1e-6)

    def test_knn_bonus_running_sq_dist_ewma_update(self, discrete_env):
        """``_running_sq_dist`` updates as decay·old + (1-decay)·mean(top-k d²)."""
        decay = 0.99
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=2),
            optimizer_params=OPT_PARAMS,
            memory_size=10,
            k=2,
            running_mean_decay=decay,
            device=DEVICE,
        )
        # Two memory points equidistant (d² = 1) from the origin query.
        epi._memories[0] = T.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=T.float32
        )
        epi._running_sq_dist = T.tensor(1.0, device=DEVICE)

        x = T.zeros((1, 2), dtype=T.float32, device=DEVICE)
        _ = epi._knn_bonus(x, T.tensor([0], dtype=T.long, device=DEVICE))
        # top-k=2 d² = [1.0, 1.0]; mean = 1.0 → new = 0.99·1 + 0.01·1 = 1.0
        assert T.allclose(
            epi._running_sq_dist.cpu(), T.tensor(1.0), atol=1e-6
        )

        # Second query: d² to (1,0) = 1, d² to (0,1) = 5  (top-2 = [1, 5])
        x2 = T.tensor([[2.0, 0.0]], dtype=T.float32, device=DEVICE)
        _ = epi._knn_bonus(x2, T.tensor([0], dtype=T.long, device=DEVICE))
        expected = T.tensor(decay * 1.0 + (1 - decay) * ((1.0 + 5.0) / 2.0))
        assert T.allclose(epi._running_sq_dist.cpu(), expected, atol=1e-6)

    def test_knn_bonus_caps_at_max_similarity(self, discrete_env):
        """sim > max_similarity² ⇒ bonus is clipped to 0.0."""
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=2),
            optimizer_params=OPT_PARAMS,
            memory_size=64,
            k=8,
            kernel_epsilon=1e-3,
            cluster_distance=8e-3,
            max_similarity=0.5,  # max_similarity² = 0.25
            running_mean_decay=0.99,
            device=DEVICE,
        )
        # Memory filled with copies of the query → every kernel = 1.
        # k=8 neighbours all at distance zero → sim = 8.0 > 0.25.
        epi._memories[0] = T.zeros((10, 2), dtype=T.float32)
        epi._running_sq_dist = T.tensor(1.0, device=DEVICE)
        x = T.zeros((1, 2), dtype=T.float32, device=DEVICE)

        bonus = epi._knn_bonus(x, T.tensor([0], dtype=T.long, device=DEVICE))
        assert T.allclose(bonus, T.zeros_like(bonus))

    def test_compute_rollout_reward_scales_with_reward_weight(self, discrete_env):
        """Cold-memory bonus = 1.0 ⇒ output = reward_weight (no normalizer)."""
        for w in (0.5, 2.0):
            epi = EpisodicNovelty(
                env=discrete_env,
                model_configs=episodic_configs(embed_dim=4),
                optimizer_params=OPT_PARAMS,
                memory_size=32,
                k=2,
                reward_weight=w,
                device=DEVICE,
            )
            batch = _random_batch(discrete_env, batch=NUM_ENVS)
            r = epi.compute_rollout_reward(
                **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
            )
            assert T.allclose(r, w * T.ones_like(r))

    def test_get_config_keys(self, epi_discrete):
        cfg = epi_discrete.get_config()
        for key in (
            "env",
            "model_configs",
            "optimizer_params",
            "memory_size",
            "k",
            "kernel_epsilon",
            "cluster_distance",
            "max_similarity",
            "running_mean_decay",
            "reward_weight",
            "reward_scheduler",
            "extrinsic_threshold",
            "reward_normalizer",
        ):
            assert key in cfg
        assert cfg["k"] == 3
        assert cfg["memory_size"] == 64

    def test_save_load_roundtrip(self, discrete_env, tmp_path):
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=8),
            optimizer_params=OPT_PARAMS,
            memory_size=64,
            k=3,
            reward_weight=0.7,
            extrinsic_threshold=2,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        # Touch all the lazy modules
        _ = epi.compute_rollout_reward(
            **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
        )
        epi.save(tmp_path)

        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)
        assert isinstance(loaded, EpisodicNovelty)
        assert loaded.k == 3
        assert loaded.memory_size == 64
        assert loaded.reward_weight == 0.7
        assert loaded.extrinsic_threshold == 2

        _to_cpu(loaded)
        _state_dicts_match(epi.state_dict(), loaded.state_dict())


# =============================================================================
# CompositeIntrinsicMotivation tests
# =============================================================================
class TestCompositeIntrinsicMotivation:
    @pytest.fixture
    def composite_components(self, discrete_env):
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            device=DEVICE,
        )
        return [icm, epi]

    @pytest.fixture
    def composite(self, discrete_env, composite_components):
        return CompositeIntrinsicMotivation(
            env=discrete_env,
            components=composite_components,
            combination_rule="additive",
            device=DEVICE,
        )

    def test_init_registers_components_as_submodules(self, composite, composite_components):
        # Each child must be a registered submodule for state_dict propagation
        named = dict(composite.named_children())
        for i, c in enumerate(composite_components):
            assert f"component_{i}" in named
            assert named[f"component_{i}"] is c

    def test_invalid_combination_rule(self, discrete_env, composite_components):
        with pytest.raises(ValueError, match="Unknown combination_rule"):
            CompositeIntrinsicMotivation(
                env=discrete_env,
                components=composite_components,
                combination_rule="not_a_rule",
                device=DEVICE,
            )

    def test_is_online_property(self, composite, composite_components):
        # EpisodicNovelty is online
        assert composite.is_online is True

        # Pure parametric composite (ICM + RND) is not online
        icm = composite_components[0]
        rnd = RND(
            env=composite.env,
            model_configs=rnd_configs(output_dim=8),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        pure = CompositeIntrinsicMotivation(
            env=composite.env,
            components=[icm, rnd],
            combination_rule="additive",
            device=DEVICE,
        )
        assert pure.is_online is False

    def test_split_components(self, composite, composite_components):
        online, parametric = composite._split_components()
        # In our fixture: index 0 = ICM (parametric), index 1 = Episodic (online)
        assert [i for i, _ in online] == [1]
        assert [i for i, _ in parametric] == [0]

    def test_compute_rollout_reward_uses_online_only(self, composite, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        r = composite.compute_rollout_reward(**batch, env_indices=env_indices)
        assert r.shape == (NUM_ENVS,)
        # First call to EpisodicNovelty with empty memory → bonus = 1.0
        assert T.allclose(r, T.ones_like(r))

    def test_compute_learn_reward_combines_with_rollout(self, composite, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        rollout = composite.compute_rollout_reward(**batch, env_indices=env_indices)
        learn = composite.compute_learn_reward(**batch, rollout_rewards=rollout)
        assert learn.shape == (NUM_ENVS,)
        assert T.isfinite(learn).all().item()

    def test_compute_learn_reward_only_parametric(self, discrete_env):
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[icm],
            combination_rule="additive",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        learn = comp.compute_learn_reward(**batch)
        assert learn.shape == (NUM_ENVS,)
        # No rollout reward → result equals ICM's learn reward scaled by composite weight
        ref = icm.compute_learn_reward(**batch)
        assert T.allclose(learn, ref * comp._scaled_reward_weight(), atol=1e-5)

    def test_compute_learn_reward_only_rollout(self, discrete_env):
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi],
            combination_rule="additive",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        rollout = comp.compute_rollout_reward(**batch, env_indices=env_indices)
        learn = comp.compute_learn_reward(**batch, rollout_rewards=rollout)
        assert T.allclose(learn, rollout * comp._scaled_reward_weight(), atol=1e-5)

    def test_compute_learn_reward_raises_when_empty(self, discrete_env):
        # Trick: build a composite with a single parametric component and call
        # compute_learn_reward with no rollout_rewards and force the parametric
        # branch to be empty by faking the split. Easier: just call with no
        # components — but the constructor allows it.
        # Build a composite where neither parametric nor online exists.
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[],
            combination_rule="additive",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        with pytest.raises(RuntimeError, match="No learn or rollout"):
            comp.compute_learn_reward(**batch, rollout_rewards=None)

    def test_train_calls_each_child(self, composite, composite_components, discrete_env):
        batch = _random_batch(discrete_env, batch=8)
        # Snapshot params per child
        snapshots = [
            [p.detach().clone() for p in c.parameters()] for c in composite_components
        ]
        loss = composite.train(**batch)
        assert isinstance(loss, T.Tensor)
        assert T.isfinite(loss).item()
        # Each child should have updated at least one parameter
        for child, snap in zip(composite_components, snapshots):
            any_changed = any(
                not T.equal(a, b) for a, b in zip(snap, child.parameters())
            )
            assert any_changed, f"{child.__class__.__name__} not updated by composite.train"

    # ----- Math correctness for combination rules -------------------------
    def test_additive_combination_default_weights_math(self, discrete_env):
        """Composite additive rollout = Σ componentᵢ_reward (default weights = 1)."""
        epi1 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=2.0,
            device=DEVICE,
        )
        epi2 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=3.0,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi1, epi2],
            combination_rule="additive",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        # Both components are on cold memory ⇒ each contributes its
        # reward_weight ⇒ additive sum = 2.0 + 3.0 = 5.0.
        r = comp.compute_rollout_reward(
            **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
        )
        assert T.allclose(r, 5.0 * T.ones_like(r))

    def test_additive_combination_with_explicit_weights_math(self, discrete_env):
        """Composite weights apply on top of per-component reward_weights."""
        epi1 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=2.0,
            device=DEVICE,
        )
        epi2 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=3.0,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi1, epi2],
            combination_rule="additive",
            combination_kwargs={"weights": [0.5, 0.25]},
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        r = comp.compute_rollout_reward(
            **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
        )
        # 0.5 * 2.0 + 0.25 * 3.0 = 1.75
        assert T.allclose(r, 1.75 * T.ones_like(r))

    def test_max_combination_math(self, discrete_env):
        epi1 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=2.0,
            device=DEVICE,
        )
        epi2 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=5.0,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi1, epi2],
            combination_rule="max",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        r = comp.compute_rollout_reward(
            **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
        )
        assert T.allclose(r, 5.0 * T.ones_like(r))

    def test_multiplicative_combination_math(self, discrete_env):
        epi1 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=2.0,
            device=DEVICE,
        )
        epi2 = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=3.0,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi1, epi2],
            combination_rule="multiplicative",
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        r = comp.compute_rollout_reward(
            **batch, env_indices=T.arange(NUM_ENVS, dtype=T.long)
        )
        # 2.0 * 3.0 = 6.0
        assert T.allclose(r, 6.0 * T.ones_like(r))

    def test_composite_learn_reward_scales_by_composite_reward_weight(
        self, discrete_env
    ):
        """Composite ``compute_learn_reward`` multiplies the final combined
        reward by the composite's own ``_scaled_reward_weight``.
        """
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(embed_dim=4),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            reward_weight=1.0,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[epi],
            combination_rule="additive",
            reward_weight=0.5,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_idx = T.arange(NUM_ENVS, dtype=T.long)
        # Cold-memory child returns 1.0 per row → rollout = 1.0
        rollout = comp.compute_rollout_reward(**batch, env_indices=env_idx)
        assert T.allclose(rollout, T.ones_like(rollout))
        # Composite scales by 0.5 ⇒ learn reward should equal 0.5 * rollout
        # (no parametric components in this composite).
        learn = comp.compute_learn_reward(**batch, rollout_rewards=rollout)
        assert T.allclose(learn, 0.5 * rollout, atol=1e-6)

    def test_on_episode_end_propagates(self, composite, composite_components, discrete_env):
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)
        # Populate EpisodicNovelty memory
        _ = composite.compute_rollout_reward(**batch, env_indices=env_indices)
        epi = composite_components[1]
        assert all(mem.shape[0] > 0 for mem in epi._memories)
        composite.on_episode_end(T.tensor([0, 1], dtype=T.long))
        assert all(mem.shape[0] == 0 for mem in epi._memories)

    def test_get_config(self, composite):
        cfg = composite.get_config()
        for key in (
            "env",
            "components",
            "combination_rule",
            "combination_kwargs",
            "reward_weight",
            "reward_scheduler",
            "extrinsic_threshold",
        ):
            assert key in cfg
        assert cfg["combination_rule"] == "additive"
        assert len(cfg["components"]) == 2

    def test_save_load_roundtrip(self, discrete_env, tmp_path):
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        rnd = RND(
            env=discrete_env,
            model_configs=rnd_configs(output_dim=8),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        comp = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[icm, rnd],
            combination_rule="additive",
            reward_weight=0.4,
            extrinsic_threshold=7,
            device=DEVICE,
        )
        # Touch lazy modules
        batch = _random_batch(discrete_env, batch=4)
        _ = icm.compute_learn_reward(**batch)
        _ = rnd.compute_learn_reward(**batch)
        comp.save(tmp_path)

        cfg_path = tmp_path / "intrinsic_motivation" / "config.json"
        assert cfg_path.exists()
        # Each child gets its own sub-folder
        comp_root = tmp_path / "intrinsic_motivation"
        assert (comp_root / "component_0_ICM" / "intrinsic_motivation").exists()
        assert (comp_root / "component_1_RND" / "intrinsic_motivation").exists()

        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)
        assert isinstance(loaded, CompositeIntrinsicMotivation)
        assert loaded.combination_rule == "additive"
        assert loaded.reward_weight == 0.4
        assert loaded.extrinsic_threshold == 7
        assert len(loaded.components) == 2
        assert isinstance(loaded.components[0], ICM)
        assert isinstance(loaded.components[1], RND)


# =============================================================================
# reward_scheduler save/load round-trip (regression, all four classes)
# =============================================================================
class TestRewardSchedulerRoundtrip:
    """Regression coverage: ``reward_scheduler`` must survive save/load.

    Every concrete ``IntrinsicMotivation`` subclass serializes its optional
    ``reward_scheduler`` under the ``'reward_scheduler'`` key in
    ``get_config()``. Each ``_load_impl`` must read that same key back. A
    mismatch (reading a different, never-written key) makes the reloaded
    module silently fall back to ``reward_scheduler=None``, which in turn
    makes ``_scaled_reward_weight()`` skip the schedule's decay entirely on
    a resumed run — with no exception raised anywhere.
    """

    @pytest.mark.parametrize(
        "kind", ["ICM", "RND", "EpisodicNovelty", "CompositeIntrinsicMotivation"]
    )
    def test_reward_scheduler_survives_save_load(self, discrete_env, tmp_path, kind):
        """A save/load round trip must preserve ``reward_scheduler`` exactly.

        Builds ``kind`` with a real ``ScheduleWrapper``, saves it to
        ``tmp_path``, reloads it via ``IntrinsicMotivation.load``, and
        checks that the reloaded module has a non-``None`` scheduler whose
        config matches the original, and that
        ``_scaled_reward_weight()`` — the value a training loop actually
        consumes — agrees before and after the round trip.

        Args:
            discrete_env: Vector CartPole env fixture.
            tmp_path: Pytest tmp directory for the save tree.
            kind: Which registered intrinsic-motivation class to exercise.
        """
        sched = ScheduleWrapper(
            schedule_type="linear",
            steps=100,
            start_value=1.0,
            end_value=0.1,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)

        if kind == "ICM":
            module = ICM(
                env=discrete_env,
                model_configs=icm_configs(),
                optimizer_params=OPT_PARAMS,
                reward_scheduler=sched,
                device=DEVICE,
            )
            # Materialize LazyLinear layers before saving.
            _ = module.compute_learn_reward(**batch)
        elif kind == "RND":
            module = RND(
                env=discrete_env,
                model_configs=rnd_configs(output_dim=16),
                optimizer_params=OPT_PARAMS,
                reward_scheduler=sched,
                device=DEVICE,
            )
            _ = module.compute_learn_reward(**batch)
        elif kind == "EpisodicNovelty":
            module = EpisodicNovelty(
                env=discrete_env,
                model_configs=episodic_configs(),
                optimizer_params=OPT_PARAMS,
                memory_size=32,
                k=2,
                reward_scheduler=sched,
                device=DEVICE,
            )
            _ = module.compute_rollout_reward(**batch, env_indices=env_indices)
        else:  # CompositeIntrinsicMotivation
            icm = ICM(
                env=discrete_env,
                model_configs=icm_configs(),
                optimizer_params=OPT_PARAMS,
                device=DEVICE,
            )
            epi = EpisodicNovelty(
                env=discrete_env,
                model_configs=episodic_configs(),
                optimizer_params=OPT_PARAMS,
                memory_size=32,
                k=2,
                device=DEVICE,
            )
            _ = icm.compute_learn_reward(**batch)
            _ = epi.compute_rollout_reward(**batch, env_indices=env_indices)
            module = CompositeIntrinsicMotivation(
                env=discrete_env,
                components=[icm, epi],
                combination_rule="additive",
                reward_scheduler=sched,
                device=DEVICE,
            )

        module.save(tmp_path)
        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)

        assert loaded.reward_scheduler is not None, (
            f"{kind}: reward_scheduler was lost on save/load round trip"
        )
        assert loaded.reward_scheduler.get_config() == sched.get_config()
        assert pytest.approx(loaded._scaled_reward_weight()) == module._scaled_reward_weight()

    @pytest.mark.parametrize(
        "kind", ["ICM", "RND", "EpisodicNovelty", "CompositeIntrinsicMotivation"]
    )
    def test_reward_scheduler_resumes_after_reload(self, discrete_env, tmp_path, kind):
        """A stepped ``reward_scheduler`` must resume mid-schedule after reload.

        Unlike ``test_reward_scheduler_survives_save_load`` above (which never
        advances the schedule, so both sides of its comparison sit at the
        unstepped factor of ``1.0``), this steps the schedule well past its
        start *before* saving. That makes the assertion provably non-vacuous:
        the captured factor is asserted to differ from ``1.0`` before the
        round trip even begins, then the post-reload factor and scaled
        reward weight are asserted to match the pre-save values exactly.

        Args:
            discrete_env: Vector CartPole env fixture.
            tmp_path: Pytest tmp directory for the save tree.
            kind: Which registered intrinsic-motivation class to exercise.
        """
        sched = ScheduleWrapper(
            schedule_type="linear",
            steps=100,
            start_value=1.0,
            end_value=0.1,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)

        if kind == "ICM":
            module = ICM(
                env=discrete_env,
                model_configs=icm_configs(),
                optimizer_params=OPT_PARAMS,
                reward_scheduler=sched,
                device=DEVICE,
            )
            # Materialize LazyLinear layers before saving.
            _ = module.compute_learn_reward(**batch)
        elif kind == "RND":
            module = RND(
                env=discrete_env,
                model_configs=rnd_configs(output_dim=16),
                optimizer_params=OPT_PARAMS,
                reward_scheduler=sched,
                device=DEVICE,
            )
            _ = module.compute_learn_reward(**batch)
        elif kind == "EpisodicNovelty":
            module = EpisodicNovelty(
                env=discrete_env,
                model_configs=episodic_configs(),
                optimizer_params=OPT_PARAMS,
                memory_size=32,
                k=2,
                reward_scheduler=sched,
                device=DEVICE,
            )
            _ = module.compute_rollout_reward(**batch, env_indices=env_indices)
        else:  # CompositeIntrinsicMotivation
            icm = ICM(
                env=discrete_env,
                model_configs=icm_configs(),
                optimizer_params=OPT_PARAMS,
                device=DEVICE,
            )
            epi = EpisodicNovelty(
                env=discrete_env,
                model_configs=episodic_configs(),
                optimizer_params=OPT_PARAMS,
                memory_size=32,
                k=2,
                device=DEVICE,
            )
            _ = icm.compute_learn_reward(**batch)
            _ = epi.compute_rollout_reward(**batch, env_indices=env_indices)
            module = CompositeIntrinsicMotivation(
                env=discrete_env,
                components=[icm, epi],
                combination_rule="additive",
                reward_scheduler=sched,
                device=DEVICE,
            )

        # Advance the schedule well past its unstepped start before saving.
        sched.step(30)
        factor_before = module.reward_scheduler.get_factor()
        weight_before = module._scaled_reward_weight()
        assert factor_before != pytest.approx(1.0), (
            f"{kind}: test setup error — schedule factor is still at its "
            "unstepped start, so this test would pass vacuously"
        )

        module.save(tmp_path)
        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)

        factor_after = loaded.reward_scheduler.get_factor()
        weight_after = loaded._scaled_reward_weight()

        assert factor_after == pytest.approx(factor_before), (
            f"{kind}: reward_scheduler did not resume its progress after reload "
            f"(before={factor_before}, after={factor_after})"
        )
        assert weight_after == pytest.approx(weight_before)

    def test_composite_child_scheduler_resumes_after_reload(self, discrete_env, tmp_path):
        """A CHILD's own ``reward_scheduler`` must resume too, not just the composite's.

        Gives one child (``ICM``) its own ``ScheduleWrapper`` while the
        composite itself carries none, steps the child's schedule, saves the
        composite, reloads it, and asserts the reloaded child's factor
        matches the pre-save value. This is what proves the recursive
        restore through ``IntrinsicMotivation.load``'s dispatcher — which
        calls ``_load_schedule_state`` once per component directory as
        ``CompositeIntrinsicMotivation._load_impl`` rebuilds each child via
        that same dispatcher — actually reaches nested components.

        Args:
            discrete_env: Vector CartPole env fixture.
            tmp_path: Pytest tmp directory for the save tree.
        """
        child_sched = ScheduleWrapper(
            schedule_type="linear",
            steps=100,
            start_value=1.0,
            end_value=0.1,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        env_indices = T.arange(NUM_ENVS, dtype=T.long)

        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            reward_scheduler=child_sched,
            device=DEVICE,
        )
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            device=DEVICE,
        )
        # Materialize LazyLinear layers on both children before saving.
        _ = icm.compute_learn_reward(**batch)
        _ = epi.compute_rollout_reward(**batch, env_indices=env_indices)

        composite = CompositeIntrinsicMotivation(
            env=discrete_env,
            components=[icm, epi],
            combination_rule="additive",
            device=DEVICE,
        )

        child_sched.step(30)
        child_factor_before = icm.reward_scheduler.get_factor()
        assert child_factor_before != pytest.approx(1.0), (
            "Test setup error — child schedule factor is still at its "
            "unstepped start, so this test would pass vacuously"
        )

        composite.save(tmp_path)
        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)

        loaded_icm = loaded.components[0]
        assert isinstance(loaded_icm, ICM)
        assert loaded_icm.reward_scheduler is not None
        assert loaded_icm.reward_scheduler.get_factor() == pytest.approx(child_factor_before), (
            "Composite reload did not resume a child component's own "
            "reward_scheduler progress"
        )

    def test_missing_schedule_state_file_is_silent_noop(self, discrete_env, tmp_path):
        """A checkpoint saved before this feature existed must still load cleanly.

        Saves a module with a stepped ``reward_scheduler``, deletes the
        ``schedule_state.pt`` file that ``_save_schedule_state`` wrote, then
        reloads. Loading must succeed without raising, the scheduler must
        still be present (rebuilt from ``config.json``), but its progress
        must be reset to the start of the schedule since no progress file
        was available to restore it. This pins the deliberate silent no-op
        documented on ``IntrinsicMotivation._load_schedule_state``.

        Args:
            discrete_env: Vector CartPole env fixture.
            tmp_path: Pytest tmp directory for the save tree.
        """
        sched = ScheduleWrapper(
            schedule_type="linear",
            steps=100,
            start_value=1.0,
            end_value=0.1,
        )
        batch = _random_batch(discrete_env, batch=NUM_ENVS)
        module = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            reward_scheduler=sched,
            device=DEVICE,
        )
        _ = module.compute_learn_reward(**batch)

        sched.step(30)
        assert sched.get_factor() != pytest.approx(1.0), (
            "Test setup error — schedule factor is still at its unstepped start"
        )

        module.save(tmp_path)
        schedule_state_path = tmp_path / "intrinsic_motivation" / "schedule_state.pt"
        assert schedule_state_path.exists()
        schedule_state_path.unlink()

        # Must not raise despite the missing progress file.
        loaded = IntrinsicMotivation.load(tmp_path, env=discrete_env)

        assert loaded.reward_scheduler is not None
        assert loaded.reward_scheduler.get_factor() == pytest.approx(1.0), (
            "Scheduler progress should have reset to the start of the "
            "schedule when schedule_state.pt is absent"
        )


# =============================================================================
# IntrinsicMotivation base class behavior (through subclasses)
# =============================================================================
class TestIntrinsicMotivationBase:
    def test_create_instance_dispatch(self, discrete_env):
        icm = IntrinsicMotivation.create_instance(
            "ICM",
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        assert isinstance(icm, ICM)

        rnd = IntrinsicMotivation.create_instance(
            "RND",
            env=discrete_env,
            model_configs=rnd_configs(output_dim=8),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        assert isinstance(rnd, RND)

        epi = IntrinsicMotivation.create_instance(
            "EpisodicNovelty",
            env=discrete_env,
            model_configs=episodic_configs(),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            device=DEVICE,
        )
        assert isinstance(epi, EpisodicNovelty)

    def test_default_zero_learn_reward_via_episodic(self, discrete_env):
        epi = EpisodicNovelty(
            env=discrete_env,
            model_configs=episodic_configs(),
            optimizer_params=OPT_PARAMS,
            memory_size=32,
            k=2,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=4)
        out = epi.compute_learn_reward(**batch)
        assert out.shape == (4,)
        assert T.allclose(out, T.zeros_like(out))

    def test_default_zero_rollout_reward_via_icm(self, discrete_env):
        # ICM does not implement compute_rollout_reward — should be the base
        # default that returns zeros.
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            device=DEVICE,
        )
        batch = _random_batch(discrete_env, batch=4)
        out = icm.compute_rollout_reward(**batch)
        assert out.shape == (4,)
        assert T.allclose(out, T.zeros_like(out))

    def test_load_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No config"):
            IntrinsicMotivation.load(tmp_path)

    def test_load_unknown_type_raises(self, tmp_path):
        import json as _json

        model_dir = tmp_path / "intrinsic_motivation"
        model_dir.mkdir(parents=True)
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            _json.dump({"type": "DoesNotExist"}, f)
        with pytest.raises(ValueError, match="Unknown intrinsic motivation type"):
            IntrinsicMotivation.load(tmp_path)

    def test_reward_normalizer_eval_mode_propagation(self, discrete_env):
        norm = RewardNorm(gamma=0.99, clip_value=5.0, device=DEVICE)
        icm = ICM(
            env=discrete_env,
            model_configs=icm_configs(),
            optimizer_params=OPT_PARAMS,
            reward_normalizer=norm,
            device=DEVICE,
        )
        icm.set_normalizers_mode("train")
        assert norm.training is True
        icm.set_normalizers_mode("eval")
        assert norm.training is False
