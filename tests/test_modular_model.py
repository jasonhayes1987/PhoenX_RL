"""Unit tests for the composite ``ModularModel`` in ``src/app/models.py``.

Uses the *real* production classes end to end (real ``GymnasiumWrapper`` envs
where an env is needed; the synthetic multi-modal / goal envs come from
``conftest.py``). Covers:

    * construction (objects and from_config), JSON-serializable configs;
    * user-visible parameter naming (``roots.<name>...`` / ``trunk...`` /
      ``branches.<role>...``);
    * input routing: per-root input_keys, declaration-order concat, error
      cases (missing keys, dict obs without roots, flat obs with input_keys);
    * goal handling for flat, dict-routed, and fallback-concat paths;
    * temporal-placement enforcement (memory layers only in the trunk);
    * per-module optimizers over disjoint params, zero_grad/step/clip subsets;
    * save/load state (weights + optimizer + scheduler) and cloning
      (full, weightless, branch subset, device);
    * name-matched ``soft_update`` with a branch-subset target;
    * hidden-state protocol: step==sequence exactness and masked mid-sequence
      resets equal manual episode splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as T

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.agent_utils import soft_update  # noqa: E402
from app.env_wrapper import GymnasiumWrapper  # noqa: E402
from app.models import (  # noqa: E402
    ContinuousQHead,
    DeterministicActorHead,
    DiscreteQHead,
    ModularModel,
    StochasticContinuousHead,
    StochasticDiscreteHead,
    SubNetwork,
    ValueHead,
    build_model,
)
from app.schedulers import ScheduleWrapper  # noqa: E402

DEVICE = "cpu"
T.manual_seed(0)
np.random.seed(0)

LC = [
    {"type": "dense", "params": {"units": 24, "kernel": "orthogonal", "kernel_params": {"gain": 1.41421356}}},
    {"type": "relu"},
]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


# =============================================================================
# Env fixtures (module-scoped; real GymnasiumWrapper)
# =============================================================================
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
def multimodal():
    env = GymnasiumWrapper(cfg="PhoenXMultiModal-v0", num_envs=2, seed=0)
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


def _ppo_style_model(env, roots=None, trunk=None, **kwargs):
    return ModularModel(
        env=env,
        roots=roots,
        trunk=trunk,
        branches={
            "policy": StochasticDiscreteHead(env, layer_config=LC, output_config=OC, device=DEVICE),
            "value": ValueHead(env, layer_config=LC, output_config=OC, device=DEVICE),
        },
        optimizer_params=OPT,
        device=DEVICE,
        **kwargs,
    )


def _shared_continuous_model(env, shared_update="critic"):
    return ModularModel(
        env=env,
        roots={"state": SubNetwork(LC, name="state", optimizer_params={"type": "SGD", "params": {"lr": 0.1}})},
        trunk=SubNetwork(LC, name="trunk"),
        branches={
            "policy": StochasticContinuousHead(env, layer_config=LC, output_config=OC,
                                               distribution="normal", device=DEVICE),
            "critic": ContinuousQHead(env, layer_config=LC, merged_config=LC,
                                      output_config=OC, device=DEVICE),
            "critic_b": ContinuousQHead(env, layer_config=LC, merged_config=LC,
                                        output_config=OC, device=DEVICE),
        },
        optimizer_params=OPT,
        shared_update=shared_update,
        device=DEVICE,
    )


def _mm_roots():
    return {
        "cnn": SubNetwork([
            {"type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "stride": 2}},
            {"type": "relu"},
            {"type": "flatten"},
        ], input_keys=["rgb"], name="cnn"),
        "state": SubNetwork(LC, input_keys=["vec"], name="state"),
    }


# =============================================================================
# Construction & naming
# =============================================================================
class TestConstruction:
    def test_branches_only(self, cartpole):
        model = _ppo_style_model(cartpole)
        assert model.roots is None and model.trunk is None
        assert set(model.optimizers) == {"branches.policy", "branches.value"}
        assert model.shared_module_names() == []

    def test_requires_branches(self, cartpole):
        with pytest.raises(ValueError, match="at least one branch"):
            ModularModel(env=cartpole, branches={}, device=DEVICE)

    def test_invalid_shared_update(self, cartpole):
        with pytest.raises(ValueError, match="shared_update"):
            _ppo_style_model(cartpole, shared_update="bogus")

    def test_param_names_are_user_visible(self, multimodal):
        model = ModularModel(
            env=multimodal,
            roots=_mm_roots(),
            trunk=SubNetwork(LC, name="trunk"),
            branches={"policy": StochasticContinuousHead(multimodal, layer_config=LC, output_config=OC,
                                                         distribution="normal", device=DEVICE),
                      "value": ValueHead(multimodal, layer_config=LC, output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        names = [n for n, _ in model.named_parameters()]
        assert any(n.startswith("roots.cnn.layers.conv2d_0.") for n in names)
        assert any(n.startswith("roots.state.layers.dense_0.") for n in names)
        assert any(n.startswith("trunk.layers.dense_0.") for n in names)
        assert any(n.startswith("branches.policy.body.layers.") for n in names)
        assert any(n.startswith("branches.policy.output_layer.policy_output_param_1.") for n in names)
        assert any(n.startswith("branches.value.output_layer.value_dense_output.") for n in names)

    def test_temporal_layers_only_in_trunk(self, pendulum):
        rec = SubNetwork([{"type": "lstm", "params": {"hidden_size": 8}}], name="rec")
        with pytest.raises(ValueError, match="only allowed in the trunk"):
            ModularModel(
                env=pendulum, roots={"rec": rec},
                branches={"value": ValueHead(pendulum, layer_config=LC, output_config=OC, device=DEVICE)},
                device=DEVICE,
            )
        with pytest.raises(ValueError, match="only allowed in the trunk"):
            ModularModel(
                env=pendulum,
                branches={"value": ValueHead(
                    pendulum,
                    layer_config=[{"type": "gru", "params": {"hidden_size": 8}}],
                    output_config=OC, device=DEVICE)},
                device=DEVICE,
            )

    def test_dry_run_error_names_module(self, pendulum):
        """A conv root on a flat vector env must fail with the root named."""
        bad_root = SubNetwork([
            {"type": "conv2d", "params": {"out_channels": 4}},
        ], name="cam")
        with pytest.raises(RuntimeError, match="root 'cam'"):
            ModularModel(
                env=pendulum, roots={"cam": bad_root},
                branches={"value": ValueHead(pendulum, layer_config=LC, output_config=OC, device=DEVICE)},
                device=DEVICE,
            )

    def test_optimizer_params_inheritance_and_override(self, pendulum):
        model = _shared_continuous_model(pendulum)
        # root declared SGD lr=0.1; others inherit model-level Adam lr=1e-3
        assert isinstance(model.optimizers["roots.state"], T.optim.SGD)
        assert model.optimizers["roots.state"].param_groups[0]["lr"] == pytest.approx(0.1)
        assert isinstance(model.optimizers["trunk"], T.optim.Adam)
        assert model.optimizers["trunk"].param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_optimizer_param_sets_are_disjoint_and_complete(self, pendulum):
        model = _shared_continuous_model(pendulum)
        seen = set()
        for name, opt in model.optimizers.items():
            for group in opt.param_groups:
                for p in group["params"]:
                    assert id(p) not in seen, f"param shared across optimizers ({name})"
                    seen.add(id(p))
        all_params = {id(p) for p in model.parameters() if p.requires_grad}
        assert seen == all_params, "some trainable params are not covered by any optimizer"

    def test_lr_scheduler_default_cloned_per_module(self, cartpole):
        sched = ScheduleWrapper(schedule_type="linear", steps=100, start_value=1e-3, end_value=1e-4)
        model = _ppo_style_model(cartpole, lr_scheduler=sched)
        assert set(model.lr_schedulers) == set(model.optimizers)
        # stepping one scheduler must not affect the other's optimizer
        lr_before = model.optimizers["branches.value"].param_groups[0]["lr"]
        model.lr_schedulers["branches.policy"].step(50)
        assert model.optimizers["branches.value"].param_groups[0]["lr"] == pytest.approx(lr_before)
        assert model.optimizers["branches.policy"].param_groups[0]["lr"] < lr_before


# =============================================================================
# Forward routing
# =============================================================================
class TestForwardRouting:
    def test_branch_selection(self, cartpole):
        model = _ppo_style_model(cartpole)
        obs = T.randn(5, 4)
        out, _ = model(obs, branches="policy")
        assert set(out.keys()) == {"policy"}
        out, _ = model(obs)
        assert set(out.keys()) == {"policy", "value"}
        with pytest.raises(KeyError, match="Unknown branch role"):
            model(obs, branches=("bogus",))

    def test_dict_obs_routing_and_isolation(self, multimodal):
        model = ModularModel(
            env=multimodal, roots=_mm_roots(), trunk=SubNetwork(LC, name="trunk"),
            branches={"value": ValueHead(multimodal, layer_config=[], output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        rgb = T.randint(0, 255, (3, 16, 16, 3), dtype=T.uint8)
        vec = T.randn(3, 7)
        base, _ = model({"rgb": rgb, "vec": vec})
        # Perturbing either modality must change the output (both are wired in)
        out_rgb, _ = model({"rgb": T.zeros_like(rgb), "vec": vec})
        out_vec, _ = model({"rgb": rgb, "vec": vec + 1.0})
        assert not T.allclose(base["value"], out_rgb["value"])
        assert not T.allclose(base["value"], out_vec["value"])

    def test_dict_obs_missing_key_raises(self, multimodal):
        model = ModularModel(
            env=multimodal, roots=_mm_roots(),
            branches={"value": ValueHead(multimodal, layer_config=[], output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        with pytest.raises(KeyError, match="input key 'vec'"):
            model({"rgb": T.randint(0, 255, (3, 16, 16, 3), dtype=T.uint8)})

    def test_dict_obs_without_roots_raises(self, cartpole):
        model = _ppo_style_model(cartpole)
        with pytest.raises(ValueError, match="Dict observations require roots"):
            model({"a": T.randn(3, 4)})

    def test_flat_obs_with_input_keys_raises(self, pendulum):
        root = SubNetwork(LC, input_keys=["vec"], name="state")
        with pytest.raises((ValueError, RuntimeError)):
            ModularModel(
                env=pendulum, roots={"state": root},
                branches={"value": ValueHead(pendulum, layer_config=[], output_config=OC, device=DEVICE)},
                device=DEVICE,
            )

    def test_root_concat_order_is_declaration_order(self, multimodal):
        """With an identity trunk and a value head whose first weight row is
        known, the concat order of root outputs is observable: swapping the
        declaration order must change the output for asymmetric weights."""
        def build(order):
            roots = _mm_roots()
            ordered = {k: roots[k] for k in order}
            return ModularModel(
                env=multimodal, roots=ordered,
                branches={"value": ValueHead(multimodal, layer_config=[], output_config=OC, device=DEVICE)},
                optimizer_params=OPT, device=DEVICE,
            )
        T.manual_seed(3)
        m1 = build(["cnn", "state"])
        T.manual_seed(3)
        m2 = build(["state", "cnn"])
        obs = {"rgb": T.randint(0, 255, (2, 16, 16, 3), dtype=T.uint8), "vec": T.randn(2, 7)}
        o1, _ = m1(obs)
        o2, _ = m2(obs)
        # Same seed, same shapes, but permuted feature order -> different output
        assert not T.allclose(o1["value"], o2["value"])

    def test_uint8_images_scaled(self, multimodal):
        """uint8 inputs are cast to float and divided by 255 before the CNN."""
        model = ModularModel(
            env=multimodal, roots=_mm_roots(),
            branches={"value": ValueHead(multimodal, layer_config=[], output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        rgb = T.full((2, 16, 16, 3), 255, dtype=T.uint8)
        vec = T.zeros(2, 7)
        out_u8, _ = model({"rgb": rgb, "vec": vec})
        out_float, _ = model({"rgb": T.ones(2, 16, 16, 3), "vec": vec})
        assert T.allclose(out_u8["value"], out_float["value"], atol=1e-6)

    def test_continuous_q_requires_action(self, pendulum):
        model = _shared_continuous_model(pendulum)
        obs = T.randn(4, 3)
        with pytest.raises(RuntimeError, match="requires an `action`"):
            model(obs, branches=("critic",))
        out, _ = model(obs, action=T.randn(4, 1), branches=("critic", "critic_b"))
        assert out["critic"].shape == (4, 1)
        assert not T.allclose(out["critic"], out["critic_b"])  # independent heads


class TestGoalHandling:
    def test_flat_goal_concat(self, goal_env):
        model = ModularModel(
            env=goal_env,
            branches={"value": ValueHead(goal_env, layer_config=LC, output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        obs = T.randn(4, 6)
        goal = T.randn(4, 3)
        out, _ = model(obs, goal=goal)
        assert out["value"].shape == (4, 1)
        out2, _ = model(obs, goal=goal + 1.0)
        assert not T.allclose(out["value"], out2["value"])  # goal is wired in
        # Missing goal changes the input width -> matmul shape error
        with pytest.raises(RuntimeError):
            model(obs)

    def test_dict_root_requests_goal_input(self, goal_env):
        roots = {
            "state": SubNetwork(LC, input_keys=["observation", "goal"], name="state"),
        }
        model = ModularModel(
            env=goal_env, roots=roots,
            branches={"value": ValueHead(goal_env, layer_config=[], output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )
        obs = {"observation": T.randn(4, 6)}
        goal = T.randn(4, 3)
        out, _ = model(obs, goal=goal)
        assert out["value"].shape == (4, 1)
        with pytest.raises((ValueError, RuntimeError), match="goal"):
            model(obs)  # goal requested but not provided


# =============================================================================
# Optimizer coordination surface
# =============================================================================
class TestOptimizerCoordination:
    def test_zero_grad_step_subsets(self, pendulum):
        model = _shared_continuous_model(pendulum)
        obs = T.randn(8, 3)
        act = T.randn(8, 1)
        before = {k: v.clone() for k, v in model.state_dict().items()}

        model.zero_grad()
        out, _ = model(obs, action=act, branches=("critic",))
        out["critic"].pow(2).mean().backward()
        # Step ONLY the critic branch + shared body
        model.step(model.branch_module_names("critic") + model.shared_module_names())

        after = model.state_dict()
        changed = {k for k in before if not T.equal(before[k], after[k])}
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("roots.state.") for k in changed)
        assert any(k.startswith("trunk.") for k in changed)
        assert not any(k.startswith("branches.policy.") for k in changed)
        assert not any(k.startswith("branches.critic_b.") for k in changed)

    def test_clip_returns_norm_over_subset(self, pendulum):
        model = _shared_continuous_model(pendulum)
        obs = T.randn(8, 3)
        model.zero_grad()
        out, _ = model(obs, branches=("policy",))
        out["policy"].log_prob(out["policy"].sample()).mean().backward()
        norm = model.clip(1e9, modules=model.branch_module_names("policy"))
        assert norm > 0.0
        # Manual norm over the same params must match
        params = [p for g in model.optimizers["branches.policy"].param_groups for p in g["params"]]
        manual = T.sqrt(sum(p.grad.pow(2).sum() for p in params if p.grad is not None))
        assert norm == pytest.approx(float(manual), rel=1e-5)
        # Actually clipping shrinks the norm to the max
        clipped_to = 0.5 * norm
        model.clip(clipped_to, modules=model.branch_module_names("policy"))
        new_norm = T.sqrt(sum(p.grad.pow(2).sum() for p in params if p.grad is not None))
        assert float(new_norm) == pytest.approx(float(clipped_to), rel=1e-4)

    def test_unknown_module_raises(self, pendulum):
        model = _shared_continuous_model(pendulum)
        with pytest.raises(KeyError, match="Unknown module"):
            model.step(["branches.nope"])


# =============================================================================
# Serialization, cloning, soft updates
# =============================================================================
class TestSerialization:
    def test_config_json_roundtrip(self, pendulum):
        model = _shared_continuous_model(pendulum)
        cfg = model.get_config()
        blob = json.dumps(cfg)  # must be JSON-serializable
        rebuilt = build_model(json.loads(blob), env=pendulum)
        assert isinstance(rebuilt, ModularModel)
        assert rebuilt.get_config() == cfg
        # identical parameter structure
        assert [n for n, _ in rebuilt.named_parameters()] == [n for n, _ in model.named_parameters()]

    def test_save_load_state_full_equality(self, pendulum, tmp_path):
        model = _shared_continuous_model(pendulum)
        # take an optimizer step so Adam state exists
        obs, act = T.randn(4, 3), T.randn(4, 1)
        model.zero_grad()
        out, _ = model(obs, action=act)
        (out["critic"].mean() + out["critic_b"].mean()
         + out["policy"].log_prob(act).mean()).backward()
        model.step()

        path = tmp_path / "model.pt"
        model.save_state(path)
        rebuilt = ModularModel.from_config(model.get_config()["config"], env=pendulum)
        rebuilt.load_state(path)

        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), rebuilt.state_dict().items()):
            assert k1 == k2 and T.equal(v1, v2), k1
        for name in model.optimizers:
            s1 = model.optimizers[name].state_dict()
            s2 = rebuilt.optimizers[name].state_dict()
            assert str(s1["param_groups"]) == str(s2["param_groups"])
            for pid, st in s1["state"].items():
                for key, val in st.items():
                    if isinstance(val, T.Tensor):
                        assert T.equal(val, s2["state"][pid][key]), (name, pid, key)

    def test_clone_full_and_weightless(self, pendulum):
        model = _shared_continuous_model(pendulum)
        clone = model.clone()
        for k, v in clone.state_dict().items():
            assert T.equal(v, model.state_dict()[k]), k
        fresh = model.clone(copy_weights=False)
        assert any(
            not T.equal(v, model.state_dict()[k])
            for k, v in fresh.state_dict().items() if v.dim() >= 2
        )

    def test_clone_branch_subset(self, pendulum):
        model = _shared_continuous_model(pendulum)
        target = model.clone(branches=["critic", "critic_b"])
        assert set(target.branches.keys()) == {"critic", "critic_b"}
        assert set(k for k in target.state_dict()) <= set(model.state_dict())
        for k, v in target.state_dict().items():
            assert T.equal(v, model.state_dict()[k]), k
        with pytest.raises(KeyError, match="Unknown branch"):
            model.clone(branches=["nope"])

    def test_soft_update_name_matched_subset(self, pendulum):
        model = _shared_continuous_model(pendulum)
        target = model.clone(branches=["critic", "critic_b"])
        # Perturb the source
        with T.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        soft_update(model, target, tau=1.0)  # full copy
        src = model.state_dict()
        for k, v in target.state_dict().items():
            assert T.allclose(v, src[k]), k
        # Partial tau
        with T.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        before = {k: v.clone() for k, v in target.state_dict().items()}
        soft_update(model, target, tau=0.25)
        src = model.state_dict()
        for k, v in target.state_dict().items():
            expected = before[k] + 0.25 * (src[k] - before[k])
            assert T.allclose(v, expected, atol=1e-6), k


# =============================================================================
# Recurrent trunk: hidden protocol + exactness
# =============================================================================
class TestRecurrentTrunk:
    @pytest.fixture()
    def rec_model(self, pendulum):
        return ModularModel(
            env=pendulum,
            roots={"state": SubNetwork(LC, name="state")},
            trunk=SubNetwork([
                {"type": "dense", "params": {"units": 16}},
                {"type": "relu"},
                {"type": "lstm", "params": {"hidden_size": 16}},
            ], name="trunk"),
            branches={"value": ValueHead(pendulum, layer_config=[], output_config=OC, device=DEVICE)},
            optimizer_params=OPT, device=DEVICE,
        )

    def test_hidden_keys_and_shapes(self, rec_model):
        assert rec_model.is_recurrent and rec_model.is_temporal
        hidden = rec_model.init_hidden(4)
        assert list(hidden.keys()) == ["trunk.lstm_2"]
        h, c = hidden["trunk.lstm_2"]
        assert h.shape == (1, 4, 16) and c.shape == (1, 4, 16)
        assert T.all(h == 0) and T.all(c == 0)

    def test_step_equals_sequence(self, rec_model):
        T.manual_seed(4)
        x = T.randn(3, 6, 3)
        hidden = rec_model.init_hidden(3)
        outs = []
        for t in range(6):
            out, hidden = rec_model(x[:, t], hidden=hidden, mode="step")
            outs.append(out["value"])
        stepwise = T.stack(outs, dim=1)
        full, final_hidden = rec_model(x, hidden=rec_model.init_hidden(3), mode="sequence")
        assert T.allclose(stepwise, full["value"], atol=1e-5)
        # final hidden state must match too
        h_step, c_step = hidden["trunk.lstm_2"]
        h_seq, c_seq = final_hidden["trunk.lstm_2"]
        assert T.allclose(h_step, h_seq, atol=1e-5)
        assert T.allclose(c_step, c_seq, atol=1e-5)

    def test_step_start_mask_resets_hidden(self, rec_model):
        x = T.randn(2, 3)
        # Warm the hidden state, then reset env 0 only
        _, hidden = rec_model(T.randn(2, 3), hidden=rec_model.init_hidden(2), mode="step")
        mask = T.tensor([True, False])
        out_reset, _ = rec_model(x, hidden=hidden, start_mask=mask, mode="step")
        out_fresh, _ = rec_model(x, hidden=rec_model.init_hidden(2), mode="step")
        out_kept, _ = rec_model(x, hidden=hidden, mode="step")
        assert T.allclose(out_reset["value"][0], out_fresh["value"][0], atol=1e-6)
        assert T.allclose(out_reset["value"][1], out_kept["value"][1], atol=1e-6)

    def test_sequence_masked_reset_equals_manual_split(self, rec_model):
        T.manual_seed(5)
        x = T.randn(2, 7, 3)
        start = T.zeros(2, 7, dtype=T.bool)
        start[:, 4] = True
        masked, _ = rec_model(x, hidden=rec_model.init_hidden(2), start_mask=start, mode="sequence")
        first, _ = rec_model(x[:, :4], hidden=rec_model.init_hidden(2), mode="sequence")
        second, _ = rec_model(x[:, 4:], hidden=rec_model.init_hidden(2), mode="sequence")
        manual = T.cat([first["value"], second["value"]], dim=1)
        assert T.allclose(masked["value"], manual, atol=1e-5)

    def test_recurrent_save_load_clone(self, rec_model, pendulum, tmp_path):
        path = tmp_path / "rec.pt"
        rec_model.save_state(path)
        rebuilt = ModularModel.from_config(rec_model.get_config()["config"], env=pendulum)
        rebuilt.load_state(path)
        x = T.randn(2, 5, 3)
        a, _ = rec_model(x, mode="sequence")
        b, _ = rebuilt(x, mode="sequence")
        assert T.allclose(a["value"], b["value"], atol=1e-6)


# =============================================================================
# The shipped canonical multi-modal config must parse and build
# =============================================================================
class TestCanonicalConfig:
    def test_multi_modal_cfg_yaml_builds(self):
        """src/Configs/multi_modal_cfg.yml (the schema reference) must load,
        decompose into modular parts, and assemble into a working PPO model
        against a matching multi-modal observation space."""
        import gymnasium as gym
        import yaml
        from app.models import modular_parts_from_config

        cfg_path = Path(__file__).resolve().parents[1] / "src" / "Configs" / "multi_modal_cfg.yml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        model_cfg = raw["agent"]["config"]["model"]

        class FakeIsaacEnv:
            single_observation_space = gym.spaces.Dict({
                "rgb": gym.spaces.Box(0, 255, (32, 32, 3), np.uint8),
                "policy": gym.spaces.Box(-np.inf, np.inf, (39,), np.float32),
            })
            observation_space = single_observation_space
            single_action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
            action_space = single_action_space
            obs_key = None
            goal_key = None

        env = FakeIsaacEnv()
        parts = modular_parts_from_config({**model_cfg, "device": "cpu"}, env)
        assert set(parts["roots"].keys()) == {"camera", "state"}
        assert parts["roots"]["camera"].input_keys == ["rgb"]
        assert parts["shared_update"] == "combined"

        model = ModularModel(
            env=env, roots=parts["roots"], trunk=parts["trunk"],
            branches=parts["branches"], optimizer_params=parts["optimizer_params"],
            shared_update=parts["shared_update"], device="cpu",
        )
        obs = {"rgb": T.randint(0, 255, (4, 32, 32, 3), dtype=T.uint8),
               "vec": None, "policy": T.randn(4, 39)}
        obs.pop("vec")
        out, _ = model(obs)
        assert out["value"].shape == (4, 1)
        assert out["policy"].sample().shape == (4, 8)
        # per-module optimizer overrides from the YAML are honored
        assert model.optimizers["roots.camera"].param_groups[0]["lr"] == pytest.approx(3.0e-4)
        assert model.optimizers["branches.value"].param_groups[0]["lr"] == pytest.approx(1.0e-4)

    def test_ppo_camera_yaml_builds_with_isaac_shaped_obs(self):
        """The Franka cube-lift camera training config
        (src/Configs/IsaacSim/franka/cube_lift/dense/ppo_camera.yml) must build
        against IsaacLab-shaped observations: channels-LAST uint8 frames from a
        TiledCamera group ('rgb') plus a 36-dim proprio group ('policy'). The
        env stub is deliberately NOT a GymnasiumWrapper — the HWC->CHW
        conversion must apply to IsaacSim-sourced images too."""
        import gymnasium as gym
        import yaml
        from app.models import modular_parts_from_config

        cfg_path = (Path(__file__).resolve().parents[1] / "src" / "Configs" / "IsaacSim"
                    / "franka" / "cube_lift" / "dense" / "ppo_camera.yml")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        model_cfg = raw["agent"]["config"]["model"]

        assert raw["env"]["config"]["enable_cameras"] is True
        assert raw["env"]["config"]["obs_key"] is None

        class FakeIsaacEnv:
            single_observation_space = gym.spaces.Dict({
                "rgb": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
                "policy": gym.spaces.Box(-np.inf, np.inf, (36,), np.float32),
            })
            observation_space = single_observation_space
            single_action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
            action_space = single_action_space
            obs_key = None
            goal_key = None

        env = FakeIsaacEnv()
        parts = modular_parts_from_config({**model_cfg, "device": "cpu"}, env)
        model = ModularModel(
            env=env, roots=parts["roots"], trunk=parts["trunk"],
            branches=parts["branches"], optimizer_params=parts["optimizer_params"],
            shared_update=parts["shared_update"], device="cpu",
        )
        obs = {"rgb": T.randint(0, 255, (4, 84, 84, 3), dtype=T.uint8),
               "policy": T.randn(4, 36)}
        out, _ = model(obs)
        assert out["value"].shape == (4, 1)
        assert out["policy"].sample().shape == (4, 8)
        # HWC->CHW happened: the first conv materialized with 3 in-channels
        # (not 84), even though the env stub is not a GymnasiumWrapper.
        conv0 = model.roots["camera"].layers["conv2d_0"]
        assert conv0.weight.shape[1] == 3

    def test_preprocess_input_channels_last_conversion(self):
        """_preprocess_input: uint8 HWC frames are scaled to [0,1] and permuted
        to CHW for image roots regardless of env wrapper type; already-CHW
        input passes through unpermuted."""
        conv_root = SubNetwork(
            layer_config=[{"type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3}}],
            input_keys=["rgb"], name="camera",
        )

        class FlatEnvStub:  # not a GymnasiumWrapper
            single_observation_space = None
            observation_space = None
            obs_key = None
            goal_key = None

        model = ModularModel.__new__(ModularModel)  # skip dry-run init
        model.env = FlatEnvStub()
        model.device = T.device("cpu")

        hwc = T.randint(0, 255, (5, 84, 84, 3), dtype=T.uint8)
        out = model._preprocess_input(hwc, conv_root)
        assert out.shape == (5, 3, 84, 84)
        assert out.dtype == T.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert T.allclose(out[0, :, 0, 0], hwc[0, 0, 0, :].float() / 255.0)

        chw = T.rand(5, 3, 84, 84)
        assert model._preprocess_input(chw, conv_root).shape == (5, 3, 84, 84)
        # grayscale (N, H, W) gains a channel dim
        gray = T.rand(5, 84, 84)
        assert model._preprocess_input(gray, conv_root).shape == (5, 1, 84, 84)

    def test_normal_head_sigma_bounded_under_extreme_features(self, pendulum):
        """The 'normal' branch clamps its pre-softplus scale like the beta
        branch: even pathological features (early-normalizer whipsaw, buffer
        dtype bugs) must yield finite, bounded sigma — never 0 (infinite
        log-probs) nor softplus overflow."""
        head = StochasticContinuousHead(
            pendulum, layer_config=list(LC), output_config=list(OC),
            optimizer_params=OPT, distribution="normal", device=DEVICE)
        head(T.randn(4, 24))  # materialize lazy layers

        for scale in (1.0, 1e3, 1e6):
            dist = head(T.randn(64, 24) * scale)
            base = dist
            while hasattr(base, "base_dist"):
                base = base.base_dist
            while hasattr(base, "dist"):
                base = base.dist
            sigma = base.scale
            assert T.isfinite(sigma).all()
            lo = T.nn.functional.softplus(T.tensor(-12.0)) + 1e-6
            hi = T.nn.functional.softplus(T.tensor(6.0)) + 1e-6
            assert (sigma >= lo - 1e-9).all() and (sigma <= hi + 1e-6).all()
            z = base.loc + sigma * T.randn_like(sigma)
            assert T.isfinite(dist.log_prob_from_z(z)).all()
