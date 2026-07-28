"""Recurrent (temporal) training tests — Phase 4 (on-policy) and Phase 5
(off-policy R2D2-style + temporal transformer).

All tests use the REAL production classes. Model-level step==sequence and
masked-reset exactness live in test_modular_model.py; this file covers the
AGENT/TRAINER-level protocol:

    * rollout hidden carry: ``act(dones=...)`` resets per-env hidden exactly
      like a manual model-level stream (agent wiring, not just model math);
    * the rollout-window snapshot: learn() recomputes the same old log-probs
      the policy produced while acting (staleness-free bookkeeping);
    * recurrent PPO / ActorCritic learn: finite metrics, every module updates,
      minibatch-of-envs partitioning;
    * Reinforce rejects temporal trunks;
    * a short end-to-end recurrent PPO training run through the real Trainer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as T


from phoenx.env_wrapper import GymnasiumWrapper
from phoenx.models import (
    ModularModel,
    StochasticDiscreteHead,
    SubNetwork,
    ValueHead,
)
from phoenx.rl_agents import ActorCritic, PPO, Reinforce

DEVICE = "cpu"
T.manual_seed(0)
np.random.seed(0)

LC = [{"type": "dense", "params": {"units": 16}}, {"type": "relu"}]
OC = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
OPT = {"type": "Adam", "params": {"lr": 1e-3}}


@pytest.fixture(scope="module")
def cartpole():
    env = GymnasiumWrapper(cfg="CartPole-v1", num_envs=4, seed=0)
    yield env
    try:
        env.close()
    except Exception:
        pass


def _recurrent_ppo(env, **kw):
    return PPO(
        roots={"state": SubNetwork(LC, name="state")},
        trunk=SubNetwork([
            {"type": "dense", "params": {"units": 16}},
            {"type": "relu"},
            {"type": "lstm", "params": {"hidden_size": 16}},
        ], name="trunk"),
        policy=StochasticDiscreteHead(env, layer_config=[], output_config=OC, device=DEVICE),
        value=ValueHead(env, layer_config=[], output_config=OC, device=DEVICE),
        optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE, **kw,
    )


def _rollout_sample(env, traj_len=6, first_step_at=None):
    """Buffer-shaped rollout sample with optional per-env episode starts."""
    E = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    first_steps = T.zeros(traj_len, E, dtype=T.bool)
    if first_step_at is not None:
        for e, t in first_step_at.items():
            first_steps[t, e] = True
    total = traj_len * E
    valid = (first_steps.reshape(-1) == 0).nonzero()
    return {
        "states": T.randn(traj_len, E, obs_dim),
        "actions": T.randint(0, 2, (traj_len, E, 1)).float(),
        "raw_actions": T.zeros(traj_len, E, 1),
        "rewards": T.randn(traj_len, E),
        "intrinsic_rewards": T.zeros(traj_len, E),
        "next_states": T.randn(traj_len, E, obs_dim),
        "terminations": T.zeros(traj_len, E, dtype=T.bool),
        "truncations": T.zeros(traj_len, E, dtype=T.bool),
        "log_probs": T.randn(traj_len, E) * 0.1,
        "first_steps": first_steps,
        "valid_indices": valid,
        "state_achieved_goals": None,
        "next_state_achieved_goals": None,
        "desired_goals": None,
    }


# =============================================================================
# Rollout hidden protocol (agent-level)
# =============================================================================
class TestRolloutHiddenProtocol:
    def test_act_carries_and_resets_hidden_like_manual_stream(self, cartpole):
        agent = _recurrent_ppo(cartpole)
        E = cartpole.num_envs
        T.manual_seed(1)
        obs_seq = T.randn(5, E, 4)
        dones_seq = T.zeros(5, E, dtype=T.bool)
        dones_seq[2, 1] = True  # env 1 resets before consuming step 2

        agent.reset_hidden()
        agent_logits = []
        for t in range(5):
            outputs = agent._rollout_forward(obs_seq[t], branches=("policy",),
                                             dones=dones_seq[t])
            agent_logits.append(outputs["policy"].logits)

        # Manual model-level stream with explicit hidden management
        hidden = agent.model.init_hidden(E)
        manual_logits = []
        for t in range(5):
            out, hidden = agent.model(obs_seq[t], branches=("policy",),
                                      hidden=hidden, start_mask=dones_seq[t], mode="step")
            hidden = agent.model.detach_hidden(hidden)
            manual_logits.append(out["policy"].logits)

        for t in range(5):
            assert T.allclose(agent_logits[t], manual_logits[t], atol=1e-6), f"step {t}"

    def test_learn_recomputes_rollout_log_probs_exactly(self, cartpole):
        """The old log-probs PPO's recurrent learn computes internally (sequence
        forward from the rollout-start snapshot) must equal the log-probs the
        policy emitted while acting — validating snapshot timing and start-mask
        semantics end to end."""
        agent = _recurrent_ppo(cartpole)
        E = cartpole.num_envs
        T.manual_seed(2)
        traj_len = 6
        obs_seq = T.randn(traj_len, E, 4)
        dones_seq = T.zeros(traj_len, E, dtype=T.bool)
        dones_seq[3, 0] = True
        dones_seq[4, 2] = True

        agent.reset_hidden()
        acted_log_probs = T.zeros(traj_len, E)
        actions = T.zeros(traj_len, E, dtype=T.long)
        with T.no_grad():
            for t in range(traj_len):
                outputs = agent._rollout_forward(obs_seq[t], branches=("policy",),
                                                 dones=dones_seq[t])
                dist = outputs["policy"]
                a = dist.sample()
                actions[t] = a
                acted_log_probs[t] = dist.log_prob(a)

        # Recompute exactly as the recurrent learn does: sequence forward from
        # the rollout-start snapshot with first_steps as the start mask.
        hidden0 = agent._rollout_start_hidden
        states_seq = obs_seq.transpose(0, 1).contiguous()
        start_mask = dones_seq.transpose(0, 1)
        with T.no_grad():
            out, _ = agent.model(states_seq, branches=("policy",), hidden=hidden0,
                                 start_mask=start_mask, mode="sequence")
            recomputed = out["policy"].log_prob(actions.transpose(0, 1))

        assert T.allclose(recomputed, acted_log_probs.transpose(0, 1), atol=1e-5)

    def test_reset_hidden_clears_state(self, cartpole):
        agent = _recurrent_ppo(cartpole)
        agent._rollout_forward(T.randn(cartpole.num_envs, 4), branches=("policy",))
        assert agent._hidden is not None
        agent.reset_hidden()
        assert agent._hidden is None and agent._rollout_start_hidden is None


# =============================================================================
# Recurrent learn()
# =============================================================================
class TestRecurrentLearn:
    def test_ppo_recurrent_learn_updates_all_modules(self, cartpole):
        agent = _recurrent_ppo(cartpole)
        sample = _rollout_sample(cartpole, traj_len=6, first_step_at={1: 2})
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample, learning_epochs=2, mini_batch_size=2)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any("lstm" in k for k in changed), "recurrent trunk did not update"
        assert any(k.startswith("roots.state.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)
        assert any(k.startswith("branches.value.") for k in changed)

    def test_ppo_recurrent_minibatch_env_partition(self, cartpole):
        """mini_batch_size is interpreted in env units: sizes that don't divide
        num_envs fall back to full-batch, valid sizes partition the envs."""
        agent = _recurrent_ppo(cartpole)
        # 4 envs, env-batches of 2 -> two updates per epoch (runs cleanly)
        metrics = agent.learn(0, _rollout_sample(cartpole, 5), learning_epochs=1, mini_batch_size=2)
        assert np.isfinite(metrics["policy_loss"])
        # 3 does not divide 4 -> falls back to all envs (still runs)
        metrics = agent.learn(0, _rollout_sample(cartpole, 5), learning_epochs=1, mini_batch_size=3)
        assert np.isfinite(metrics["policy_loss"])

    def test_ppo_advance_rollout_window(self, cartpole):
        agent = _recurrent_ppo(cartpole)
        agent.reset_hidden()
        # act through a few steps so a live hidden exists
        for _ in range(3):
            agent.act(T.randn(cartpole.num_envs, 4), context="train",
                      dones=T.zeros(cartpole.num_envs, dtype=T.bool))
        live_hidden = agent._hidden
        agent.learn(0, _rollout_sample(cartpole, 4), learning_epochs=1, mini_batch_size=4)
        # after learn, the live hidden became the next window's start snapshot
        for key, value in agent._rollout_start_hidden.items():
            if isinstance(value, tuple):
                for a, b in zip(value, live_hidden[key]):
                    assert T.equal(a, b)
            else:
                assert T.equal(value, live_hidden[key])

    def test_actor_critic_recurrent_learn(self, cartpole):
        agent = ActorCritic(
            roots={"state": SubNetwork(LC, name="state")},
            trunk=SubNetwork([{"type": "gru", "params": {"hidden_size": 12}}], name="trunk"),
            policy=StochasticDiscreteHead(cartpole, layer_config=[], output_config=OC, device=DEVICE),
            value=ValueHead(cartpole, layer_config=[], output_config=OC, device=DEVICE),
            optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE,
        )
        sample = _rollout_sample(cartpole, traj_len=6, first_step_at={0: 3})
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any("gru" in k for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)

    def test_reinforce_rejects_temporal_trunk(self, cartpole):
        with pytest.raises(NotImplementedError, match="Reinforce"):
            Reinforce(
                trunk=SubNetwork([{"type": "lstm", "params": {"hidden_size": 8}}], name="trunk"),
                policy=StochasticDiscreteHead(cartpole, layer_config=LC, output_config=OC, device=DEVICE),
                device=DEVICE,
            )


# =============================================================================
# Phase 5: off-policy recurrence (R2D2 stored state + burn-in)
# =============================================================================
from phoenx.buffer import ReplayBuffer
from phoenx.env_wrapper import Action
from phoenx.models import ContinuousQHead, DeterministicActorHead, StochasticContinuousHead
from phoenx.rl_agents import SAC, TD3


@pytest.fixture(scope="module")
def pendulum_nstep():
    env = GymnasiumWrapper(
        cfg="Pendulum-v1", num_envs=2, seed=0,
        wrappers=[{"type": "VectorNStepReward", "params": {"n": 3}}],
    )
    yield env
    try:
        env.close()
    except Exception:
        pass


def _recurrent_sac(env, **kw):
    return SAC(
        roots={"state": SubNetwork(LC, name="state")},
        trunk=SubNetwork([{"type": "lstm", "params": {"hidden_size": 12}}], name="trunk"),
        policy=StochasticContinuousHead(env, layer_config=[], output_config=OC,
                                        distribution="normal", device=DEVICE),
        critic=ContinuousQHead(env, layer_config=[], merged_config=LC,
                               output_config=OC, device=DEVICE),
        optimizer_params=OPT, auto_entropy_tuning=False, device=DEVICE, **kw,
    )


class TestOffPolicyRecurrent:
    def _drive(self, env, agent, buf, steps=14):
        obs = env.reset(seed=31)
        agent.reset_hidden()
        prev_done = T.zeros(env.num_envs, dtype=T.bool)
        nstep = env._find_nstep_wrapper()
        for step_i in range(steps):
            action = agent.act(obs.states, context="train", step=step_i, warmup=-1,
                               dones=prev_done)
            nstep.set_action(action)
            next_obs = env.step(action.actions)
            buf.record(next_obs, prev_observation=obs, actions=action, prev_dones=prev_done)
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs
        return obs

    def test_action_carries_pre_step_hidden(self, pendulum_nstep):
        agent = _recurrent_sac(pendulum_nstep)
        agent.reset_hidden()
        obs = pendulum_nstep.reset(seed=33)
        action = agent.act(obs.states, context="train", step=10, warmup=-1,
                           dones=T.zeros(2, dtype=T.bool))
        assert action.hidden is not None
        # First step: pre-step hidden is the zero init, flattened batch-first
        for key, value in action.hidden.items():
            assert value.shape[0] == 2  # batch-first
            assert T.all(value == 0)
        # Second step: nonzero (carried) hidden
        action2 = agent.act(obs.states, context="train", step=11, warmup=-1,
                            dones=T.zeros(2, dtype=T.bool))
        assert any(v.abs().sum() > 0 for v in action2.hidden.values())

    def test_stored_initial_hidden_matches_rollout_reference(self, pendulum_nstep):
        """The initial_hidden each emitted window carries must equal the hidden
        the agent actually used at that window's FIRST step (naive per-step
        reference record)."""
        agent = _recurrent_sac(pendulum_nstep)
        agent.reset_hidden()
        obs = pendulum_nstep.reset(seed=37)
        prev_done = T.zeros(2, dtype=T.bool)
        nstep = pendulum_nstep._find_nstep_wrapper()
        hidden_log = []  # per step: {key: (E, layers, H)}
        n = 3
        checked = 0
        for step_i in range(10):
            action = agent.act(obs.states, context="train", step=step_i, warmup=-1,
                               dones=prev_done)
            hidden_log.append({k: v.clone() for k, v in action.hidden.items()})
            nstep.set_action(action)
            next_obs = pendulum_nstep.step(action.actions)
            traj = next_obs.n_step_trajectory
            if traj is not None and "initial_hidden" in traj:
                lengths = traj["trajectory_lengths"]
                for row in range(min(len(lengths), 2)):  # main rows = env order
                    L = int(lengths[row])
                    first_step_idx = step_i - L + 1  # window covers last L steps
                    for key, buf_val in traj["initial_hidden"].items():
                        expected = hidden_log[first_step_idx][key][row]
                        assert T.allclose(buf_val[row].cpu(), expected.cpu(), atol=1e-6), (
                            f"step {step_i} row {row} key {key}"
                        )
                    checked += 1
            prev_done = T.logical_or(next_obs.terminations, next_obs.truncations).clone()
            obs = next_obs
        assert checked >= 6

    def test_recurrent_sac_learn_end_to_end(self, pendulum_nstep):
        agent = _recurrent_sac(pendulum_nstep)
        buf = ReplayBuffer(env=pendulum_nstep, buffer_size=64, N=3, device=DEVICE)
        self._drive(pendulum_nstep, agent, buf)
        assert buf.initial_hidden is not None  # stored state allocated
        sample = buf.sample(8)
        assert "initial_hidden" in sample
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any("lstm" in k for k in changed), "recurrent trunk did not update"
        assert any(k.startswith("branches.critic.") for k in changed)
        assert any(k.startswith("branches.policy.") for k in changed)

    def test_recurrent_td3_learn_with_burn_in(self, pendulum_nstep):
        agent = TD3(
            roots={"state": SubNetwork(LC, name="state")},
            trunk=SubNetwork([{"type": "gru", "params": {"hidden_size": 12}}], name="trunk"),
            policy=DeterministicActorHead(pendulum_nstep, layer_config=[], output_config=OC, device=DEVICE),
            critic=ContinuousQHead(pendulum_nstep, layer_config=[], merged_config=LC,
                                   output_config=OC, device=DEVICE),
            optimizer_params=OPT, N=3, recurrent_burn_in=1,
            policy_update_delay=1, device=DEVICE,
        )
        from phoenx.noise import NormalNoise
        agent.noise = NormalNoise(mean=0.0, stddev=0.1, device=DEVICE)
        agent.target_noise = NormalNoise(mean=0.0, stddev=0.2, device=DEVICE)
        buf = ReplayBuffer(env=pendulum_nstep, buffer_size=64, N=3, device=DEVICE)
        self._drive(pendulum_nstep, agent, buf)
        sample = buf.sample(8)
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), key
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any("gru" in k for k in changed)

    def test_burn_in_must_be_less_than_n(self, pendulum_nstep):
        with pytest.raises(ValueError, match="recurrent_burn_in"):
            TD3(
                policy=DeterministicActorHead(pendulum_nstep, layer_config=LC, output_config=OC, device=DEVICE),
                critic=ContinuousQHead(pendulum_nstep, layer_config=LC, merged_config=LC,
                                       output_config=OC, device=DEVICE),
                N=3, recurrent_burn_in=3, device=DEVICE,
            )

    def test_hidden_tensor_roundtrip(self, pendulum_nstep):
        agent = _recurrent_sac(pendulum_nstep)
        hidden = agent.model.init_hidden(4)
        # populate with real values
        out, hidden = agent.model(T.randn(4, 3), branches=("policy",), hidden=hidden)
        flat = agent.model.hidden_to_tensors(hidden)
        rebuilt = agent.model.hidden_from_tensors(flat)
        for key, value in hidden.items():
            if isinstance(value, tuple):
                for a, b in zip(value, rebuilt[key]):
                    assert T.allclose(a, b, atol=1e-7)
            else:
                assert T.allclose(value, rebuilt[key], atol=1e-7)


# =============================================================================
# Phase 5: causal-transformer context-window inference
# =============================================================================
class TestCausalContextInference:
    @pytest.fixture()
    def causal_ppo(self, cartpole):
        return PPO(
            roots={"state": SubNetwork(LC, name="state")},
            trunk=SubNetwork([
                {"type": "dense", "params": {"units": 16}},
                {"type": "unflatten", "params": {"dim": -1, "sizes": [2, 8]}},
                {"type": "flatten", "params": {"start_dim": -2}},
                {"type": "transformer_encoder",
                 "params": {"d_model": 16, "nhead": 2, "causal": True, "dropout": 0.0}},
            ], name="trunk"),
            policy=StochasticDiscreteHead(cartpole, layer_config=[], output_config=OC, device=DEVICE),
            value=ValueHead(cartpole, layer_config=[], output_config=OC, device=DEVICE),
            optimizer_params=OPT, auto_entropy_tuning=False,
            context_length=4, device=DEVICE,
        )

    def test_rolling_context_equals_full_sequence_last_step(self, causal_ppo, cartpole):
        """While the history fits in the context window, the rolling-window
        forward must equal a full-sequence forward's last position."""
        agent = causal_ppo
        agent.model.eval()
        E = cartpole.num_envs
        T.manual_seed(9)
        obs_seq = T.randn(4, E, 4)
        dones = T.zeros(E, dtype=T.bool)

        agent.reset_hidden()
        rolling = []
        with T.no_grad():
            for t in range(4):
                outputs = agent._rollout_forward(obs_seq[t], branches=("policy",), dones=dones)
                rolling.append(outputs["policy"].logits)

        with T.no_grad():
            for t in range(4):
                window = obs_seq[: t + 1].transpose(0, 1).contiguous()  # (E, t+1, 4)
                start_mask = T.zeros(E, t + 1, dtype=T.bool)
                start_mask[:, 0] = True
                full = agent.model.forward_context(window, branches=("policy",),
                                                   start_mask=start_mask)
                assert T.allclose(rolling[t], full["policy"].logits, atol=1e-5), f"t={t}"

    def test_context_window_truncates_and_resets_on_done(self, causal_ppo, cartpole):
        agent = causal_ppo
        agent.model.eval()
        E = cartpole.num_envs
        T.manual_seed(10)
        obs_seq = T.randn(7, E, 4)

        agent.reset_hidden()
        with T.no_grad():
            for t in range(6):
                dones = T.zeros(E, dtype=T.bool)
                if t == 5:
                    dones[0] = True  # env 0 starts a new episode at step 5
                agent._rollout_forward(obs_seq[t], branches=("policy",), dones=dones)
            # window is capped at context_length
            assert len(agent._ctx_obs) == 4
            outputs = agent._rollout_forward(obs_seq[6], branches=("policy",),
                                             dones=T.zeros(E, dtype=T.bool))

            # Env 0's episode restarted at t=5: its output must equal running
            # only steps 5..6 (segment masking blocks earlier context).
            window = obs_seq[5:7, 0:1].transpose(0, 1).contiguous()
            sm = T.zeros(1, 2, dtype=T.bool)
            sm[:, 0] = True
            ref = agent.model.forward_context(window, branches=("policy",), start_mask=sm)
            assert T.allclose(outputs["policy"].logits[0:1], ref["policy"].logits, atol=1e-5)

    def test_causal_ppo_recurrent_learn_path(self, causal_ppo, cartpole):
        """Causal trunks train through the Phase-4 sequence path (stateless)."""
        agent = causal_ppo
        E = cartpole.num_envs
        traj_len = 5
        sample = {
            "states": T.randn(traj_len, E, 4),
            "actions": T.randint(0, 2, (traj_len, E, 1)).float(),
            "raw_actions": T.zeros(traj_len, E, 1),
            "rewards": T.randn(traj_len, E),
            "intrinsic_rewards": T.zeros(traj_len, E),
            "next_states": T.randn(traj_len, E, 4),
            "terminations": T.zeros(traj_len, E, dtype=T.bool),
            "truncations": T.zeros(traj_len, E, dtype=T.bool),
            "log_probs": T.randn(traj_len, E) * 0.1,
            "first_steps": T.zeros(traj_len, E, dtype=T.bool),
            "valid_indices": T.arange(traj_len * E).unsqueeze(-1),
            "state_achieved_goals": None,
            "next_state_achieved_goals": None,
            "desired_goals": None,
        }
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        metrics = agent.learn(0, sample, learning_epochs=1, mini_batch_size=E)
        assert np.isfinite(metrics["policy_loss"])
        changed = {k for k in before if not T.equal(before[k], agent.model.state_dict()[k])}
        assert any("transformer_encoder" in k for k in changed)


# =============================================================================
# End-to-end recurrent PPO through the real Trainer
# =============================================================================
class TestRecurrentTrainerIntegration:
    def test_recurrent_ppo_short_training_run(self, tmp_path):
        from phoenx.builder import build_trainer_from_config

        dev = "cuda" if T.cuda.is_available() else "cpu"
        dense16 = {"type": "dense", "params": {"units": 16}}
        relu = {"type": "relu"}
        out = [{"type": "dense", "params": {"kernel": "orthogonal", "kernel_params": {"gain": 0.01}}}]
        config = {
            "save_dir": str(tmp_path) + "/",
            "log_level": "ERROR",
            "schedule": {
                "stop_unit": "timestep", "stop_units": 64, "learn_every_unit": "timestep",
                "learn_every": 16, "updates_per_learn": 1, "batch_size": 1,
                "mini_batch_size": 2, "learning_epochs": 2, "warmup_steps": 0, "seed": 7,
            },
            "agent": {"type": "PPO", "config": {
                "name": "PPO",
                "model": {"type": "ModularModel", "config": {
                    "roots": {"state": {"layer_config": [dense16, relu]}},
                    "trunk": {"layer_config": [{"type": "lstm", "params": {"hidden_size": 16}}]},
                    "branches": {
                        "policy": {"type": "StochasticDiscreteHead",
                                   "config": {"layer_config": [], "output_config": out, "device": dev}},
                        "value": {"type": "ValueHead",
                                  "config": {"layer_config": [], "output_config": out, "device": dev}},
                    },
                    "optimizer_params": {"type": "Adam", "params": {"lr": 3e-4}},
                    "shared_update": "combined",
                    "device": dev,
                }},
                "discount": 0.99, "auto_entropy_tuning": False,
                "policy_grad_clip": 1.0, "value_grad_clip": 1.0,
                "device": dev, "log_level": "ERROR",
            }},
            "env": {"type": "gymnasium", "config": {
                "cfg": "CartPole-v1", "num_envs": 2, "obs_key": None, "goal_key": None,
                "ach_goal_key": None, "wrappers": [], "render_mode": None, "seed": 7}},
            "buffer": {"type": "RolloutBuffer", "config": {"buffer_size": 8}},
        }
        trainer = build_trainer_from_config(config)
        assert trainer.agent.model.is_recurrent
        trainer.train()  # must complete without error (learn fires 4 times)
        assert trainer._step >= 64
        trainer.env.close()
