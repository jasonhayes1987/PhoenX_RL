"""Gradient-ownership and golden-equivalence tests for ``ModularModel``.

This is the mathematically critical suite backing the refactor's core
invariant: **every shared (roots/trunk) parameter receives exactly one
coordinated update per learn step**.

Covers (real production classes only):

    * ``detach_shared=True`` leaves every shared parameter's grad empty while
      still training the requested branch (SAC-AE / DrQ-v2 rule);
    * the off-policy update pattern: critic loss steps critic+shared, the
      detached policy step touches only the policy branch;
    * the on-policy combined step (one backward, step all per-module SGD
      optimizers) equals a hand-computed reference update to numerical
      precision — proving per-module optimizers over disjoint params are
      equivalent to a single optimizer;
    * with NO shared modules, combined-backward + step-all is numerically
      identical to the legacy two-backward/two-step pattern (Adam);
    * golden equivalence: a branches-only composite loaded with a legacy
      model's weights produces bit-identical outputs for every legacy model
      class (incl. a goal-conditioned case).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch as T

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.env_wrapper import GymnasiumWrapper  # noqa: E402
from app.models import (  # noqa: E402
    ActorModel,
    ContinuousCritic,
    ContinuousQHead,
    DeterministicActorHead,
    DiscreteCritic,
    DiscreteQHead,
    ModularModel,
    StochasticContinuousHead,
    StochasticContinuousPolicy,
    StochasticDiscreteHead,
    StochasticDiscretePolicy,
    SubNetwork,
    ValueHead,
    ValueModel,
    map_legacy_state_dict,
)

DEVICE = "cpu"
T.manual_seed(0)
np.random.seed(0)

LC = [
    {"type": "dense", "params": {"units": 24, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
    {"type": "dense", "params": {"units": 16, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]


@pytest.fixture(scope="module")
def cartpole():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def pendulum():
    env = GymnasiumWrapper(cfg="Pendulum-v1", num_envs=2, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def goal_env():
    env = GymnasiumWrapper(
        cfg="PhoenXGoal-v0", num_envs=2, seed=0,
        obs_key="observation", goal_key="desired_goal", ach_goal_key="achieved_goal",
    )
    yield env
    try:
        env.close()
    except Exception:
        pass


def _shared_model(env, opt=None):
    return ModularModel(
        env=env,
        roots={"state": SubNetwork(LC, name="state")},
        trunk=SubNetwork(LC, name="trunk"),
        branches={
            "policy": StochasticContinuousHead(env, layer_config=LC, output_config=OC,
                                               distribution="normal", device=DEVICE),
            "value": ValueHead(env, layer_config=LC, output_config=OC, device=DEVICE),
            "critic": ContinuousQHead(env, layer_config=LC, merged_config=LC,
                                      output_config=OC, device=DEVICE),
        },
        optimizer_params=opt or {"type": "Adam", "params": {"lr": 1e-3}},
        shared_update="critic",
        device=DEVICE,
    )


def _grads_of(model, module_names):
    grads = []
    for name in module_names:
        for group in model.optimizers[name].param_groups:
            for p in group["params"]:
                grads.append(p.grad)
    return grads


# =============================================================================
# 1. detach_shared isolation
# =============================================================================
class TestDetachShared:
    def test_detached_policy_loss_never_reaches_shared(self, pendulum):
        model = _shared_model(pendulum)
        obs = T.randn(8, 3)
        model.zero_grad()
        out, _ = model(obs, branches=("policy",), detach_shared=True)
        actions, z = out["policy"].rsample_with_z()
        loss = -out["policy"].log_prob_from_z(z).mean()
        loss.backward()

        for g in _grads_of(model, model.shared_module_names()):
            assert g is None or T.all(g == 0), "shared parameter received gradient despite detach"
        policy_grads = _grads_of(model, model.branch_module_names("policy"))
        assert any(g is not None and g.abs().sum() > 0 for g in policy_grads), \
            "policy branch got no gradient"

    def test_without_detach_shared_receives_gradient(self, pendulum):
        model = _shared_model(pendulum)
        obs = T.randn(8, 3)
        model.zero_grad()
        out, _ = model(obs, branches=("policy",), detach_shared=False)
        actions, z = out["policy"].rsample_with_z()
        (-out["policy"].log_prob_from_z(z).mean()).backward()
        shared_grads = _grads_of(model, model.shared_module_names())
        assert any(g is not None and g.abs().sum() > 0 for g in shared_grads), \
            "shared parameters expected gradients without detach"

    def test_detach_is_noop_when_no_shared_modules(self, pendulum):
        model = ModularModel(
            env=pendulum,
            branches={"value": ValueHead(pendulum, layer_config=LC, output_config=OC, device=DEVICE)},
            device=DEVICE,
        )
        obs = T.randn(4, 3)
        a, _ = model(obs, detach_shared=True)
        b, _ = model(obs, detach_shared=False)
        assert T.equal(a["value"], b["value"])


# =============================================================================
# 2. Off-policy update pattern (critic owns the shared body)
# =============================================================================
class TestOffPolicyPattern:
    def test_full_sac_style_update_ownership(self, pendulum):
        model = _shared_model(pendulum)
        obs, act = T.randn(16, 3), T.randn(16, 1)
        target_q = T.randn(16, 1)

        snap = {k: v.clone() for k, v in model.state_dict().items()}

        # critic update: grads flow through roots+trunk
        model.zero_grad()
        out, _ = model(obs, action=act, branches=("critic",))
        (out["critic"] - target_q).pow(2).mean().backward()
        model.step(model.branch_module_names("critic") + model.shared_module_names())

        after_critic = {k: v.clone() for k, v in model.state_dict().items()}
        changed = {k for k in snap if not T.equal(snap[k], after_critic[k])}
        assert any(k.startswith("roots.") for k in changed)
        assert any(k.startswith("trunk.") for k in changed)
        assert any(k.startswith("branches.critic.") for k in changed)
        assert not any(k.startswith("branches.policy.") for k in changed)
        assert not any(k.startswith("branches.value.") for k in changed)

        # policy update: detached; steps only the policy branch
        model.zero_grad()
        out, _ = model(obs, branches=("policy",), detach_shared=True)
        _, z = out["policy"].rsample_with_z()
        (-out["policy"].log_prob_from_z(z).mean()).backward()
        model.step(model.branch_module_names("policy"))

        final = model.state_dict()
        changed2 = {k for k in after_critic if not T.equal(after_critic[k], final[k])}
        assert changed2 and all(k.startswith("branches.policy.") for k in changed2), (
            f"policy step leaked outside the policy branch: "
            f"{[k for k in changed2 if not k.startswith('branches.policy.')]}"
        )


# =============================================================================
# 3. On-policy combined step == hand-computed single-optimizer reference
# =============================================================================
class TestCombinedStepReference:
    def test_combined_sgd_step_matches_manual_update(self, pendulum):
        lr = 0.05
        model = _shared_model(pendulum, opt={"type": "SGD", "params": {"lr": lr}})
        obs = T.randn(12, 3)
        fixed_action = T.rand(12, 1) * 2 - 1  # inside Pendulum's [-2, 2]
        returns = T.randn(12, 1)

        params = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]

        out, _ = model(obs, branches=("policy", "value"))
        z = out["policy"].z_from_action(fixed_action)
        policy_loss = -out["policy"].log_prob_from_z(z).mean()
        value_loss = (out["value"] - returns).pow(2).mean()
        total = policy_loss + value_loss

        # Reference gradients of the combined loss w.r.t. every parameter.
        ref_grads = T.autograd.grad(total, params, retain_graph=True, allow_unused=True)

        model.zero_grad()
        total.backward()
        model.step()  # every per-module optimizer, once

        for p, w0, g in zip(params, before, ref_grads):
            expected = w0 if g is None else w0 - lr * g
            assert T.allclose(p.detach(), expected, atol=1e-6), \
                "combined per-module step deviates from single-optimizer SGD reference"

    def test_no_double_update_of_shared_params(self, pendulum):
        """Stepping all optimizers once must apply each shared gradient exactly
        once — verified by comparing against the SGD reference above with BOTH
        losses contributing to the shared body (their gradients sum, they are
        not applied twice)."""
        lr = 0.1
        model = ModularModel(
            env=pendulum,
            roots={"state": SubNetwork([{"type": "dense", "params": {"units": 8}}], name="state")},
            trunk=None,
            branches={
                "value": ValueHead(pendulum, layer_config=[], output_config=OC, device=DEVICE),
                "value_b": ValueHead(pendulum, layer_config=[], output_config=OC, device=DEVICE),
            },
            optimizer_params={"type": "SGD", "params": {"lr": lr}},
            device=DEVICE,
        )
        obs = T.randn(6, 3)
        shared_params = [p for g in model.optimizers["roots.state"].param_groups for p in g["params"]]
        w0 = [p.detach().clone() for p in shared_params]

        out, _ = model(obs)
        loss_a = out["value"].mean()
        loss_b = out["value_b"].mean()
        ga = T.autograd.grad(loss_a, shared_params, retain_graph=True)
        gb = T.autograd.grad(loss_b, shared_params, retain_graph=True)

        model.zero_grad()
        (loss_a + loss_b).backward()
        model.step()

        for p, w, a, b in zip(shared_params, w0, ga, gb):
            expected = w - lr * (a + b)  # summed once, applied once
            assert T.allclose(p.detach(), expected, atol=1e-6)


# =============================================================================
# 4. No-sharing degenerate case: combined == legacy sequential updates
# =============================================================================
class TestLegacyEquivalentUpdates:
    def test_combined_equals_sequential_when_disjoint(self, cartpole):
        def build():
            return ModularModel(
                env=cartpole,
                branches={
                    "policy": StochasticDiscreteHead(cartpole, layer_config=LC, output_config=OC, device=DEVICE),
                    "value": ValueHead(cartpole, layer_config=LC, output_config=OC, device=DEVICE),
                },
                optimizer_params={"type": "Adam", "params": {"lr": 1e-3}},
                device=DEVICE,
            )

        model_a = build()
        model_b = build()
        model_b.load_state_dict(model_a.state_dict())  # identical weights

        obs = T.randn(10, 4)
        actions = T.randint(0, 2, (10,))
        returns = T.randn(10, 1)

        def losses(model):
            out, _ = model(obs, branches=("policy", "value"))
            pl = -(out["policy"].log_prob(actions)).mean()
            vl = (out["value"] - returns).pow(2).mean()
            return pl, vl

        # A: legacy pattern — separate backward + separate steps
        model_a.zero_grad()
        pl, vl = losses(model_a)
        pl.backward()
        vl.backward()
        model_a.step(model_a.branch_module_names("policy"))
        model_a.step(model_a.branch_module_names("value"))

        # B: combined pattern — one backward + step all
        model_b.zero_grad()
        pl, vl = losses(model_b)
        (pl + vl).backward()
        model_b.step()

        for (k1, v1), (k2, v2) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
            assert k1 == k2
            assert T.allclose(v1, v2, atol=1e-7), f"{k1} diverged between update patterns"


# =============================================================================
# 5. Golden equivalence: legacy Model classes vs branches-only composite
# =============================================================================
def _composite_with(env, role, head):
    return ModularModel(env=env, branches={role: head}, device=DEVICE)


class TestGoldenEquivalence:
    def test_value_model(self, pendulum):
        legacy = ValueModel(pendulum, layer_config=LC, output_config=OC, device=DEVICE)
        comp = _composite_with(
            pendulum, "value", ValueHead(pendulum, layer_config=LC, output_config=OC, device=DEVICE)
        )
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "value"))
        x = T.randn(16, 3)
        expected = legacy(x)
        out, _ = comp(x)
        assert T.allclose(out["value"], expected, atol=1e-6)

    def test_stochastic_continuous_policy(self, pendulum):
        legacy = StochasticContinuousPolicy(
            pendulum, layer_config=LC, output_config=OC, distribution="normal", device=DEVICE)
        comp = _composite_with(
            pendulum, "policy",
            StochasticContinuousHead(pendulum, layer_config=LC, output_config=OC,
                                     distribution="normal", device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "policy"))
        x = T.randn(16, 3)
        d_legacy = legacy(x)
        d_comp, _ = comp(x)
        d_comp = d_comp["policy"]
        act = T.rand(16, 1) * 2 - 1
        z = d_legacy.z_from_action(act)
        assert T.allclose(d_legacy.log_prob_from_z(z), d_comp.log_prob_from_z(z), atol=1e-6)
        m1, z1 = d_legacy.mean_with_z()
        m2, z2 = d_comp.mean_with_z()
        assert T.allclose(m1, m2, atol=1e-6) and T.allclose(z1, z2, atol=1e-6)
        # get_mean_actions parity (head API mirrors the legacy Model API)
        assert T.allclose(
            legacy.get_mean_actions(d_legacy),
            comp.branches["policy"].get_mean_actions(d_comp),
            atol=1e-6,
        )

    @pytest.mark.parametrize("distribution", ["beta", "kumaraswamy"])
    def test_stochastic_continuous_policy_bounded_dists(self, pendulum, distribution):
        legacy = StochasticContinuousPolicy(
            pendulum, layer_config=LC, output_config=OC, distribution=distribution, device=DEVICE)
        comp = _composite_with(
            pendulum, "policy",
            StochasticContinuousHead(pendulum, layer_config=LC, output_config=OC,
                                     distribution=distribution, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "policy"))
        x = T.randn(8, 3)
        d1 = legacy(x)
        d2, _ = comp(x)
        d2 = d2["policy"]
        act = T.rand(8, 1) * 2 - 1
        assert T.allclose(d1.log_prob(act), d2.log_prob(act), atol=1e-5)

    def test_stochastic_discrete_policy(self, cartpole):
        legacy = StochasticDiscretePolicy(
            cartpole, layer_config=LC, output_config=OC, temperature=1.7, device=DEVICE)
        comp = _composite_with(
            cartpole, "policy",
            StochasticDiscreteHead(cartpole, layer_config=LC, output_config=OC,
                                   temperature=1.7, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "policy"))
        x = T.randn(16, 4)
        d1 = legacy(x)
        d2, _ = comp(x)
        d2 = d2["policy"]
        assert T.allclose(d1.logits, d2.logits, atol=1e-6)

    def test_actor_model(self, pendulum):
        legacy = ActorModel(pendulum, layer_config=LC, output_config=OC, device=DEVICE)
        comp = _composite_with(
            pendulum, "policy",
            DeterministicActorHead(pendulum, layer_config=LC, output_config=OC, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "policy"))
        x = T.randn(16, 3)
        mu1, pi1 = legacy(x)
        (mu2, pi2) = comp(x)[0]["policy"]
        assert T.allclose(mu1, mu2, atol=1e-6)
        assert T.allclose(pi1, pi2, atol=1e-6)

    def test_continuous_critic(self, pendulum):
        legacy = ContinuousCritic(pendulum, layer_config=LC, merged_config=LC,
                                  output_config=OC, device=DEVICE)
        comp = _composite_with(
            pendulum, "critic",
            ContinuousQHead(pendulum, layer_config=LC, merged_config=LC,
                            output_config=OC, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "critic"))
        x, a = T.randn(16, 3), T.randn(16, 1)
        expected = legacy(x, a)
        out, _ = comp(x, action=a)
        assert T.allclose(out["critic"], expected, atol=1e-6)

    def test_discrete_critic(self, cartpole):
        legacy = DiscreteCritic(cartpole, layer_config=LC, output_config=OC, device=DEVICE)
        comp = _composite_with(
            cartpole, "critic",
            DiscreteQHead(cartpole, layer_config=LC, output_config=OC, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "critic"))
        x = T.randn(16, 4)
        expected = legacy(x)
        out, _ = comp(x)
        assert T.allclose(out["critic"], expected, atol=1e-6)

    def test_goal_conditioned_value(self, goal_env):
        legacy = ValueModel(goal_env, layer_config=LC, output_config=OC, device=DEVICE)
        comp = _composite_with(
            goal_env, "value", ValueHead(goal_env, layer_config=LC, output_config=OC, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "value"))
        x = T.randn(8, 6)
        goal = T.randn(8, 3)
        expected = legacy(x, goal)
        out, _ = comp(x, goal=goal)
        assert T.allclose(out["value"], expected, atol=1e-6)

    def test_goal_conditioned_continuous_critic(self, goal_env):
        legacy = ContinuousCritic(goal_env, layer_config=LC, merged_config=LC,
                                  output_config=OC, device=DEVICE)
        comp = _composite_with(
            goal_env, "critic",
            ContinuousQHead(goal_env, layer_config=LC, merged_config=LC,
                            output_config=OC, device=DEVICE))
        comp.load_state_dict(map_legacy_state_dict(legacy.state_dict(), "critic"))
        x, a, g = T.randn(8, 6), T.randn(8, 2), T.randn(8, 3)
        expected = legacy(x, a, g)
        out, _ = comp(x, action=a, goal=g)
        assert T.allclose(out["critic"], expected, atol=1e-6)
