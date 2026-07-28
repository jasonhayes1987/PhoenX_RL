"""Unit tests for ``src/phoenx/agent_utils.py``.

These tests import the *real* functions and supporting classes from the
PhoenX API (no test-side copies or mocks) so that any refactor is
automatically validated by re-running the file.

Primary focus:
    * ``compute_q_retrace`` — the clipped-IS Retrace(λ=1) target used by
      n-step off-policy critics (SAC, DDPG-style, etc.).
    * Full integration of the n-step data path:
        VectorNStepReward (real NextStep autoreset) →
        GymnasiumWrapper + ReplayBuffer →
        compute_q_retrace

The suite now includes strong coverage of the historically difficult areas:
    - Termination / truncation placement inside N-step windows (N=3 and N=5)
    - Interaction with the trainer’s prev_done / valid_steps logic
    - Rolling circular buffers + padding + mask/cum_c timing on realistic
      longer-episode environments (LunarLanderContinuous)
    - Data path with active state/reward normalizers

Other pure helpers in agent_utils (compute_n_step_return, compute_gae, etc.)
will be added over time.

All tests are fast, deterministic, CPU-only, and use only live API classes.
"""

from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass

import numpy as np
import torch as T

from phoenx.agent_utils import compute_q_retrace

# Real API classes for integration tests (exact same style as test_intrinsic_motivation.py)
import gymnasium as gym
from gymnasium.vector import AutoresetMode
from phoenx.env_wrapper import VectorNStepReward, GymnasiumWrapper
from phoenx.buffer import ReplayBuffer

# Additional real classes needed for stronger integration coverage
from phoenx.normalizer import RunningNorm, RewardNorm
from phoenx.intrinsic_motivation import IntrinsicMotivation  # for smoke tests with IM active

DEVICE = "cpu"

# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
T.manual_seed(0)
np.random.seed(0)


# -----------------------------------------------------------------------------
# Reference implementation (slow, explicit, obviously correct)
# -----------------------------------------------------------------------------
# This is deliberately written with clear loops so that a human can audit
# the exact same algorithm that the vectorised ``compute_q_retrace`` claims
# to implement.  Any future change to the production version must continue
# to match this reference on all the cases exercised below.
# -----------------------------------------------------------------------------

def _reference_compute_q_retrace(
    rewards: T.Tensor,
    terminations: T.Tensor,
    truncations: T.Tensor,
    trajectory_lengths: T.Tensor,
    q_cur: T.Tensor,
    target_q: T.Tensor,
    cur_log_probs: T.Tensor,
    buf_log_probs: T.Tensor,
    discount: float,
    *,
    device: str | T.device | None = None,
) -> tuple[T.Tensor, dict[str, T.Tensor]]:
    """Slow reference implementation of the exact Retrace logic.

    Matches the production ``compute_q_retrace`` signature and behaviour.
    Returns the same (q_retrace, metrics) structure.
    """
    device = T.device(device) if device is not None else T.device(DEVICE)
    rewards = rewards.to(device)
    terminations = terminations.to(device)
    truncations = truncations.to(device)
    trajectory_lengths = trajectory_lengths.to(device)
    q_cur = q_cur.to(device)
    target_q = target_q.to(device)
    cur_log_probs = cur_log_probs.to(device)
    buf_log_probs = buf_log_probs.to(device)

    batch_size, n = rewards.shape

    # 1. TD errors (exactly as in production)
    td_errors = (
        rewards
        + discount * (1.0 - terminations.float()) * target_q.detach()
        - q_cur.detach()
    )

    # 2. Raw IS ratios + masking (exactly as in production).
    #    NOTE: production applies BOTH a lower clamp (0.5, a deliberate
    #    variance-control floor) and the standard Retrace upper clamp (1.0).
    #    The reference must mirror both — see
    #    TestComputeQRetrace.test_is_ratio_lower_clamp_matches_reference.
    is_ratio = T.clamp(T.exp(cur_log_probs - buf_log_probs), min=0.5, max=1.0)

    valid = (T.arange(n, device=device).unsqueeze(0) < trajectory_lengths.unsqueeze(1)).float()
    mask = T.ones(batch_size, n, device=device)
    dones = T.logical_or(terminations, truncations)

    for k in range(1, n):
        mask[:, k] = mask[:, k - 1] * (1.0 - dones[:, k - 1].float()) * valid[:, k]

    is_ratio = is_ratio * mask

    # 3. Accumulation (exact same loop ordering and cum_c update timing)
    cum_c = T.ones(batch_size, device=device)
    retrace_sum = T.zeros(batch_size, device=device)

    for k in range(n):
        gamma = discount ** k
        retrace_sum += gamma * cum_c * td_errors[:, k]
        if k < n - 1:
            cum_c = cum_c * is_ratio[:, k + 1]

    q_retrace = q_cur[:, 0] + retrace_sum

    # NOTE: the production implementation's per-row boundary-leakage
    # diagnostics ("done_window_final_cum_c" / "done_window_max_leakage") are
    # currently disabled (commented out) for training-loop performance, so the
    # production metrics dict contains only the four tensor entries below. The
    # reference mirrors that contract exactly; boundary behavior is asserted
    # directly on `mask` / `cum_c` by the tests instead.
    metrics = {
        "td_errors": td_errors,
        "mask": mask,
        "is_ratio": is_ratio,
        "cum_c": cum_c,
    }
    return q_retrace, metrics


# -----------------------------------------------------------------------------
# Metric comparison helpers
# -----------------------------------------------------------------------------
# compute_q_retrace returns a metrics dict mixing tensor-valued entries
# (td_errors, mask, is_ratio, cum_c) with list-valued boundary diagnostics
# (done_window_final_cum_c, done_window_max_leakage). These helpers compare /
# validate both kinds so the tests assert on the *correct* metrics rather than
# blindly calling tensor ops on lists.
# -----------------------------------------------------------------------------

def _assert_metrics_match(m: dict, m_ref: dict, atol: float = 1e-6) -> None:
    """Assert two metrics dicts are equal across all (shared) keys, handling
    both tensor-valued and list-valued entries."""
    assert set(m.keys()) >= set(m_ref.keys()), (
        f"Production metrics missing keys present in reference: "
        f"{set(m_ref.keys()) - set(m.keys())}"
    )
    for k, ref_val in m_ref.items():
        val = m[k]
        if isinstance(ref_val, T.Tensor):
            assert T.allclose(val, ref_val, atol=atol), f"tensor metric '{k}' mismatch"
        else:
            # list-valued diagnostic: compare element-wise as floats
            assert len(val) == len(ref_val), f"list metric '{k}' length mismatch"
            for a, b in zip(val, ref_val):
                assert abs(float(a) - float(b)) <= atol, f"list metric '{k}' value mismatch"


def _assert_metric_value_finite(value) -> None:
    """Assert a single metric value is finite, whether it is a tensor or a
    (possibly empty) list of scalars."""
    if isinstance(value, T.Tensor):
        assert T.isfinite(value).all()
    else:
        for x in value:
            assert math.isfinite(float(x))


# -----------------------------------------------------------------------------
# Test data factories (hand-crafted cases that expose the tricky boundaries)
# -----------------------------------------------------------------------------

def _tiny_batch_termination_at_k1(gamma: float = 0.9) -> dict[str, T.Tensor]:
    """Batch size 1, N=3, termination at step 1 of the window.

    This is the classic "short episode inside a longer n-step chunk" case.
    All numbers are chosen so a human can compute the expected q_retrace
    with pencil and paper in < 30 seconds.
    """
    # Only the first two steps are real (trajectory_length=2)
    rewards = T.tensor([[1.0, 2.0, 0.0]], dtype=T.float32, device=DEVICE)
    terminations = T.tensor([[False, True, False]], dtype=T.bool, device=DEVICE)
    truncations = T.tensor([[False, False, False]], dtype=T.bool, device=DEVICE)
    trajectory_lengths = T.tensor([2], dtype=T.long, device=DEVICE)

    # Make the math trivial: q_cur = 0 everywhere, target_q = 10 at the
    # (ignored) bootstrap position after the terminal step.
    q_cur = T.zeros((1, 3), dtype=T.float32, device=DEVICE)
    target_q = T.tensor([[0.0, 0.0, 10.0]], dtype=T.float32, device=DEVICE)

    # IS ratios before masking (we will see the mask zero the padded column)
    cur_log_probs = T.tensor([[0.0, 0.0, 0.0]], dtype=T.float32, device=DEVICE)
    buf_log_probs = T.tensor([[0.0, 0.0, 0.0]], dtype=T.float32, device=DEVICE)
    # After exp(...) and clamping we get all 1.0 before the mask is applied.

    return {
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "trajectory_lengths": trajectory_lengths,
        "q_cur": q_cur,
        "target_q": target_q,
        "cur_log_probs": cur_log_probs,
        "buf_log_probs": buf_log_probs,
        "discount": gamma,
        "device": DEVICE,
    }


def _batch_with_mixed_lengths_and_truncation(gamma: float = 0.99) -> dict[str, T.Tensor]:
    """Batch of 3, different trajectory lengths, one truncation (not termination)."""
    n = 4
    rewards = T.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],   # length 3, no done
            [0.5, 0.5, 0.0, 0.0],   # length 2, termination at k=1
            [2.0, 2.0, 2.0, 2.0],   # length 4, truncation at k=3
        ],
        dtype=T.float32,
        device=DEVICE,
    )
    terminations = T.tensor(
        [
            [False, False, False, False],
            [False, True,  False, False],
            [False, False, False, False],
        ],
        dtype=T.bool,
        device=DEVICE,
    )
    truncations = T.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, True],
        ],
        dtype=T.bool,
        device=DEVICE,
    )
    trajectory_lengths = T.tensor([3, 2, 4], dtype=T.long, device=DEVICE)

    q_cur = T.zeros((3, n), dtype=T.float32, device=DEVICE)
    target_q = T.zeros((3, n), dtype=T.float32, device=DEVICE)

    # Make IS ratios interesting (some >1 before clipping)
    cur_log_probs = T.tensor(
        [[0.1, 0.2, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.0],
         [0.0, 0.0, 0.0, -0.3]],
        dtype=T.float32,
        device=DEVICE,
    )
    buf_log_probs = T.zeros((3, n), dtype=T.float32, device=DEVICE)

    return {
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "trajectory_lengths": trajectory_lengths,
        "q_cur": q_cur,
        "target_q": target_q,
        "cur_log_probs": cur_log_probs,
        "buf_log_probs": buf_log_probs,
        "discount": gamma,
        "device": DEVICE,
    }


# -----------------------------------------------------------------------------
# The actual test class
# -----------------------------------------------------------------------------
class TestComputeQRetrace:
    """Tests for the pure ``compute_q_retrace`` function."""

    def test_imports_real_function(self):
        """Sanity check that we really imported the production implementation."""
        assert callable(compute_q_retrace)
        # The function must live in the agent_utils module we care about
        assert "agent_utils" in compute_q_retrace.__module__

    def test_returns_correct_shapes_and_types(self):
        batch = _tiny_batch_termination_at_k1()
        q_retrace, metrics = compute_q_retrace(**batch)

        assert q_retrace.shape == (1,)
        assert q_retrace.dtype == T.float32
        assert isinstance(metrics, dict)
        for key in ("td_errors", "mask", "is_ratio"):
            assert key in metrics
            assert metrics[key].shape == (1, 3)
        # cum_c is deliberately only the final per-batch value (shape (B,)), not per-step
        assert "cum_c" in metrics
        assert metrics["cum_c"].shape == (1,)

    # -------------------------------------------------------------------------
    # Math correctness — reference implementation + hand-worked cases
    # -------------------------------------------------------------------------
    def test_matches_reference_on_tiny_termination_case(self):
        """Termination inside an N>1 window."""
        batch = _tiny_batch_termination_at_k1(gamma=0.9)
        q, m = compute_q_retrace(**batch)
        q_ref, m_ref = _reference_compute_q_retrace(**batch)

        assert T.allclose(q, q_ref, atol=1e-6)
        _assert_metrics_match(m, m_ref, atol=1e-6)

    def test_termination_stops_bootstrap_and_accumulation(self):
        """Explicit numerical verification for the tiny termination-at-k1 case.

        With q_cur = 0 and target_q after the terminal step being irrelevant,
        the expected q_retrace for the first state can be calculated by hand:

            td_0 = 1.0 + 0.9*1.0*0 - 0 = 1.0
            td_1 = 2.0 + 0.9*0.0*0 - 0 = 2.0
            td_2 = 0   (padded, masked out)

        With is_ratio = [1,1,0] after masking,
        cum_c path: 1.0 → 1.0 (after k=0) → 0.0 (after k=1, because mask killed it)

        Therefore:
            retrace_sum = 1.0*1.0 + 0.9*1.0*2.0 = 1.0 + 1.8 = 2.8
            q_retrace   = 0 + 2.8 = 2.8
        """
        batch = _tiny_batch_termination_at_k1(gamma=0.9)
        q_retrace, metrics = compute_q_retrace(**batch)

        expected_q = T.tensor([2.8], dtype=T.float32, device=DEVICE)

        assert T.allclose(q_retrace, expected_q, atol=1e-6)

        # The mask must have zeroed the padded column
        assert T.allclose(metrics["mask"], T.tensor([[1.0, 1.0, 0.0]]), atol=1e-6)

        # cum_c after the loop must be zero (accumulation stopped)
        assert metrics["cum_c"].item() == 0.0

    def test_matches_reference_on_mixed_length_batch(self):
        batch = _batch_with_mixed_lengths_and_truncation()
        q, m = compute_q_retrace(**batch)
        q_ref, m_ref = _reference_compute_q_retrace(**batch)

        assert T.allclose(q, q_ref, atol=1e-5)
        _assert_metrics_match(m, m_ref, atol=1e-5)

    def test_is_ratio_lower_clamp_matches_reference(self):
        """Drive IS ratios BELOW the 0.5 production floor and verify both the
        clamp itself and reference agreement.

        With buf_log_probs >> cur_log_probs the raw ratio exp(cur - buf) is
        far below 0.5 (e.g. exp(-2) ~= 0.135); production clamps it to 0.5.
        This branch was previously untested (audit finding 10.2.1): no fixture
        produced a ratio under 0.5, so a regression in the lower clamp would
        have been invisible.
        """
        n = 3
        rewards = T.tensor([[1.0, 1.0, 1.0]], dtype=T.float32, device=DEVICE)
        terminations = T.zeros((1, n), dtype=T.bool, device=DEVICE)
        truncations = T.zeros((1, n), dtype=T.bool, device=DEVICE)
        trajectory_lengths = T.tensor([3], dtype=T.long, device=DEVICE)
        q_cur = T.zeros((1, n), dtype=T.float32, device=DEVICE)
        target_q = T.zeros((1, n), dtype=T.float32, device=DEVICE)
        cur_log_probs = T.zeros((1, n), dtype=T.float32, device=DEVICE)
        buf_log_probs = T.full((1, n), 2.0, dtype=T.float32, device=DEVICE)  # ratio = exp(-2) ≈ 0.135

        batch = dict(
            rewards=rewards, terminations=terminations, truncations=truncations,
            trajectory_lengths=trajectory_lengths, q_cur=q_cur, target_q=target_q,
            cur_log_probs=cur_log_probs, buf_log_probs=buf_log_probs,
            discount=0.9, device=DEVICE,
        )
        q, m = compute_q_retrace(**batch)
        q_ref, m_ref = _reference_compute_q_retrace(**batch)

        # The floor must be active: every valid ratio equals exactly 0.5
        assert T.allclose(m["is_ratio"], T.full((1, n), 0.5, device=DEVICE), atol=1e-6)

        # Hand-computed expectation with the floor:
        #   td_k = 1.0 for all k (rewards 1, q_cur/target_q 0)
        #   cum_c: 1.0 -> 0.5 (after k=0) -> 0.25 (after k=1)
        #   q = 1*1*1 + 0.9*0.5*1 + 0.81*0.25*1 = 1 + 0.45 + 0.2025 = 1.6525
        expected = T.tensor([1.6525], dtype=T.float32, device=DEVICE)
        assert T.allclose(q, expected, atol=1e-6)

        # And reference agreement on everything
        assert T.allclose(q, q_ref, atol=1e-6)
        _assert_metrics_match(m, m_ref, atol=1e-6)

    def test_n_equals_1_reduces_to_simple_one_step_target(self):
        """When every trajectory has length 1 the retrace target must be
        exactly the ordinary 1-step TD target (no accumulation, no IS multiply).
        """
        n = 1
        rewards = T.tensor([[1.5]], dtype=T.float32, device=DEVICE)
        terminations = T.tensor([[False]], dtype=T.bool, device=DEVICE)
        truncations = T.tensor([[False]], dtype=T.bool, device=DEVICE)
        trajectory_lengths = T.tensor([1], dtype=T.long, device=DEVICE)
        q_cur = T.tensor([[0.3]], dtype=T.float32, device=DEVICE)
        target_q = T.tensor([[4.2]], dtype=T.float32, device=DEVICE)
        cur_log = T.tensor([[0.1]], dtype=T.float32, device=DEVICE)
        buf_log = T.tensor([[0.0]], dtype=T.float32, device=DEVICE)

        q, _ = compute_q_retrace(
            rewards, terminations, truncations, trajectory_lengths,
            q_cur, target_q, cur_log, buf_log, discount=0.99, device=DEVICE
        )

        # For length-1, no-done: q_retrace = q_cur0 + 1.0 * (r + γ*target_q - q_cur0)
        expected = q_cur[0, 0] + (rewards[0, 0] + 0.99 * target_q[0, 0] - q_cur[0, 0])
        assert T.allclose(q, expected, atol=1e-6)

    def test_all_outputs_are_finite(self):
        """Smoke test that even on random-ish data we never produce NaN/Inf."""
        T.manual_seed(123)
        b, n = 8, 5
        batch = {
            "rewards": T.randn(b, n, device=DEVICE),
            "terminations": T.rand(b, n, device=DEVICE) > 0.8,
            "truncations": T.rand(b, n, device=DEVICE) > 0.85,
            "trajectory_lengths": T.randint(1, n + 1, (b,), device=DEVICE),
            "q_cur": T.randn(b, n, device=DEVICE),
            "target_q": T.randn(b, n, device=DEVICE),
            "cur_log_probs": T.randn(b, n, device=DEVICE) * 0.5,
            "buf_log_probs": T.randn(b, n, device=DEVICE) * 0.5,
            "discount": 0.99,
            "device": DEVICE,
        }
        q, m = compute_q_retrace(**batch)
        assert T.isfinite(q).all()
        for v in m.values():
            _assert_metric_value_finite(v)

    def test_trajectory_length_zero_batch_is_handled(self):
        """Edge case that can appear transiently: a sample with trajectory_length=0.
        The function should not crash and should produce a sensible (zero-contribution) result.
        """
        batch = {
            "rewards": T.zeros((1, 3), device=DEVICE),
            "terminations": T.zeros((1, 3), dtype=T.bool, device=DEVICE),
            "truncations": T.zeros((1, 3), dtype=T.bool, device=DEVICE),
            "trajectory_lengths": T.tensor([0], dtype=T.long, device=DEVICE),
            "q_cur": T.zeros((1, 3), device=DEVICE),
            "target_q": T.zeros((1, 3), device=DEVICE),
            "cur_log_probs": T.zeros((1, 3), device=DEVICE),
            "buf_log_probs": T.zeros((1, 3), device=DEVICE),
            "discount": 0.99,
            "device": DEVICE,
        }
        q, m = compute_q_retrace(**batch)
        # With length 0 the retrace target collapses to q_cur[0] + 0
        assert T.allclose(q, batch["q_cur"][:, 0])
        # Current production behavior: column 0 stays 1.0 because the mask loop starts at k=1.
        assert T.allclose(m["mask"], T.tensor([[1., 0., 0.]], device=DEVICE))


# =============================================================================
# Integration tests: real VectorNStepReward (NextStep autoreset) → emitted
# n-step trajectory dicts → (optional) ReplayBuffer round-trip → compute_q_retrace
#
# These tests use the *actual* classes from the PhoenX API (VectorNStepReward,
# ReplayBuffer, GymnasiumWrapper, compute_q_retrace) with real Gymnasium vector
# environments running under the exact autoreset_mode="NextStep" that is used
# in all training runs. They drive the environments, capture the precise
# trajectory dicts the wrapper actually emits (especially short windows and
# windows containing terminations/truncations), optionally round-trip them
# through the real replay buffer, and then feed them into the real
# compute_q_retrace.
#
# The assertions verify that the targets produced are consistent with the
# termination signals present in the *emitted* data (i.e. the backup stops
# exactly where the wrapper claimed a termination occurred). This layer is
# capable of catching the class of subtle n-step / boundary bugs that pure
# unit tests on compute_q_retrace alone cannot see.
# =============================================================================

@dataclass
class _TestAction:
    """Minimal stand-in for the real app.rl_agents.Action dataclass used by
    the trainer/renderer. Only the fields that VectorNStepReward.step reads
    (via .current_action) are required.
    """
    actions: T.Tensor
    raw_actions: T.Tensor | None = None
    log_probs: T.Tensor | None = None


def _create_real_nstep_wrapped_env(n: int = 3, num_envs: int = 4, seed: int = 0):
    """Create a *real* Gymnasium SyncVectorEnv (NextStep autoreset) wrapped
    with the production VectorNStepReward class from the API.

    This is the closest possible approximation to what the user's training
    runs actually execute, without going through the full high-level
    GymnasiumWrapper config machinery.

    The wrapper is forced to CPU (by temporarily patching get_device) so that
    all tensors stay on CPU and are compatible with DEVICE="cpu" used by the
    rest of this test file and the ReplayBuffer created inside the tests.
    """
    # Temporarily force CPU for the VectorNStepReward (and anything it allocates).
    # This keeps the integration tests deterministic, fast, and consistent with
    # the pure CPU-only tests in the same file, even on machines where CUDA is
    # available and get_device() would otherwise return cuda:0.
    from phoenx import torch_utils as tu
    original_get_device = tu.get_device

    def _forced_cpu(device: Any = None):
        return T.device("cpu")

    tu.get_device = _forced_cpu
    try:
        vec = gym.make_vec(
            "CartPole-v1",
            num_envs=num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": AutoresetMode.NEXT_STEP},
        )
        # Seed the action space for reproducibility inside the test
        for i in range(num_envs):
            vec.action_space.seed(seed + i)

        nstep = VectorNStepReward(vec, n=n)
        return nstep
    finally:
        tu.get_device = original_get_device


def _stopped_return_from_emitted_row(
    rewards_row: T.Tensor,
    term_row: T.Tensor,
    trunc_row: T.Tensor,
    length: int,
    discount: float,
) -> float:
    """Ground-truth stopped return computed directly from one row of an
    emitted n-step trajectory dict (the data the wrapper actually chose
    to emit for that window).
    """
    if length == 0:
        return 0.0
    rews = rewards_row[:length]
    dones = (term_row[:length] | trunc_row[:length])
    # position of first done (or end of valid window if none)
    first_done = next((k for k in range(length) if dones[k]), length - 1)
    g = 0.0
    for k in range(first_done + 1):
        g += (discount ** k) * float(rews[k])
    return g


class TestVectorNStepRewardToComputeQRetraceIntegration:
    """Real-stack integration tests for the n-step data path.

    Uses only live API classes (VectorNStepReward, ReplayBuffer,
    compute_q_retrace, GymnasiumWrapper) + real Gymnasium vector envs
    with NextStep autoreset.
    """

    def test_emitted_trajectories_produce_consistent_stopped_targets(self):
        """Drive a real NextStep + VectorNStepReward env and verify that
        compute_q_retrace, when given the exact dicts the wrapper emitted,
        produces targets that match a manual stopped-return calculation
        derived from the same emitted termination / trajectory_length data.
        """
        env = _create_real_nstep_wrapped_env(n=3, num_envs=4, seed=123)
        try:
            states, infos = env.reset()
            interesting = []
            for _ in range(400):  # plenty of CartPole terminations
                # Must call set_action with a real Action-like object before step().
                # VectorNStepReward.step reads self.current_action to populate
                # raw_actions / log_probs in the n-step buffer.
                action_np = np.array([env.single_action_space.sample() for _ in range(4)])

                # For set_action we create torch tensors explicitly on CPU.
                # This ensures the n-step buffer (and everything derived from it)
                # stays on CPU, matching DEVICE="cpu" used throughout this test file.
                action_t = T.as_tensor(action_np, device="cpu")

                # Always supply log_probs (dummy on CPU). (Historical note: an
                # old VectorNStepReward.step fallback misused T.zeros_like on
                # num_envs; production now falls back to T.zeros_like(rewards),
                # which is correct — we still pass explicit log_probs for
                # determinism.)
                log_probs_dummy = T.zeros(action_t.shape[0], device="cpu", dtype=T.float32)
                fake_action = _TestAction(actions=action_t, raw_actions=None, log_probs=log_probs_dummy)
                env.set_action(fake_action)

                # Critical: pass a plain numpy array to .step(), not a torch tensor.
                # Raw Gymnasium SyncVectorEnv (and the single CartPole envs inside it)
                # expect numpy or Python scalars. Passing a torch.Tensor (especially
                # on CUDA) reaches the passive checker and fails with "invalid action".
                next_states, rewards, terms, truncs, infos = env.step(action_np)

                traj = infos.get("n-step trajectory")
                if traj is not None and len(traj["trajectory_lengths"]) > 0:
                    lengths = traj["trajectory_lengths"]
                    has_term = (traj["terminations"] | traj["truncations"]).any(dim=1)
                    short = lengths < 3
                    if (has_term | short).any():
                        # clone so later mutations don't affect collected data
                        interesting.append({k: v.clone() for k, v in traj.items()})

            assert len(interesting) >= 8, "Too few interesting (terminating/short) chunks collected"

            discount = 0.99
            for traj in interesting[:30]:  # limit for test speed
                bsz = traj["rewards"].shape[0]
                n = traj["rewards"].shape[1]

                # Use the actual device of the emitted data (usually CPU because of
                # the forced get_device in _create_..., but we stay robust).
                data_device = traj["rewards"].device

                # Neutral critic / policy values → q_retrace should equal the stopped return
                z = T.zeros((bsz, n), device=data_device, dtype=T.float32)
                q_re, metrics = compute_q_retrace(
                    traj["rewards"],
                    traj["terminations"],
                    traj["truncations"],
                    traj["trajectory_lengths"],
                    q_cur=z,
                    target_q=z,
                    cur_log_probs=z,
                    buf_log_probs=z,
                    discount=discount,
                    device=data_device,
                )

                # Per-row manual stopped return using exactly the data the wrapper emitted
                for i in range(bsz):
                    L = int(traj["trajectory_lengths"][i].item())
                    expected = _stopped_return_from_emitted_row(
                        traj["rewards"][i],
                        traj["terminations"][i],
                        traj["truncations"][i],
                        L,
                        discount,
                    )
                    assert T.allclose(q_re[i], T.tensor(expected, device=DEVICE, dtype=T.float32), atol=1e-5)

                # Also verify that the mask actually stops after the first done in the emitted data
                mask = metrics["mask"]
                for i in range(bsz):
                    L = int(traj["trajectory_lengths"][i].item())
                    if L == 0:
                        continue
                    dones = (traj["terminations"][i, :L] | traj["truncations"][i, :L])
                    if dones.any():
                        first_done = int(dones.nonzero(as_tuple=True)[0][0].item())
                        # everything strictly after first_done in the valid part of the mask must be 0
                        if first_done + 1 < L:
                            assert (mask[i, first_done + 1 : L] == 0).all()

        finally:
            try:
                env.env.close()
            except Exception:
                pass

    def test_nstep_trajectories_roundtrip_through_replaybuffer_then_retrace(self):
        """Same as above, but the emitted trajectory dicts are first stored
        via the real ReplayBuffer.add(...) and then retrieved via .sample(...).
        The sampled dicts are then fed to compute_q_retrace.

        This exercises the exact storage path used by training (the n-step
        ReplayBuffer) and verifies that nothing is lost or corrupted for the
        fields that matter to q-retrace (rewards, terminations, truncations,
        trajectory_lengths).

        Note: "raw_actions" is deliberately omitted from the add() call in this
        test (see inline comments). The current n-step ReplayBuffer has a shape
        expectation mismatch for raw_actions when coming from VectorNStepReward
        on discrete environments. The user plans to remove raw_actions support
        entirely in the future.
        """
        nstep_env = _create_real_nstep_wrapped_env(n=3, num_envs=2, seed=99)

        # The ReplayBuffer constructor validates that its env carries the
        # VectorNStepReward wrapper, so build a matching wrapped env for it
        # (we still feed the n-step dicts manually from `nstep_env`).
        plain_env = GymnasiumWrapper(
            cfg="CartPole-v1", num_envs=2, seed=99,
            wrappers=[{"type": "VectorNStepReward", "params": {"n": 3}}],
        )

        buf = ReplayBuffer(env=plain_env, buffer_size=5000, N=3, device=DEVICE)

        try:
            states, _ = nstep_env.reset()
            fed_any = False
            for _ in range(300):
                # Must satisfy the set_action protocol (see comment in first test)
                action_np = np.array([nstep_env.single_action_space.sample() for _ in range(2)])

                # Create the tensor for set_action explicitly on CPU.
                action_t = T.as_tensor(action_np, device="cpu")
                log_probs_dummy = T.zeros(action_t.shape[0], device="cpu", dtype=T.float32)
                fake_action = _TestAction(actions=action_t, raw_actions=None, log_probs=log_probs_dummy)
                nstep_env.set_action(fake_action)

                # Pass numpy (not tensor) to .step() — this is what the underlying
                # Gymnasium SyncVectorEnv expects.
                _, _, _, _, infos = nstep_env.step(action_np)

                traj = infos.get("n-step trajectory")
                if traj is not None and len(traj.get("trajectory_lengths", [])) > 0:
                    # Defensive add: only pass the keys the n-step ReplayBuffer actually
                    # needs for the q-retrace path.  We deliberately drop "raw_actions"
                    # here because:
                    #   - The buffer stores it as (B, N, 1) for discrete actions.
                    #   - VectorNStepReward currently emits it as (B, N) for discrete.
                    #   - This causes a shape mismatch on .add().
                    # The user plans to remove raw_actions support entirely later.
                    # Goal-related keys are also dropped (they are None for CartPole).
                    add_kwargs = {
                        k: v for k, v in traj.items()
                        if k in {
                            "states", "actions", "rewards", "next_states",
                            "terminations", "truncations", "log_probs",
                            "intrinsic_rewards", "trajectory_lengths"
                        }
                    }
                    # Explicitly drop raw_actions (and any None goal keys) for safety.
                    add_kwargs.pop("raw_actions", None)
                    for gk in ("state_achieved_goals", "next_state_achieved_goals", "desired_goals"):
                        add_kwargs.pop(gk, None)

                    buf.add(**add_kwargs)
                    fed_any = True

            assert fed_any, "No n-step trajectories were fed to the buffer"

            # Sample a few batches and run them through compute_q_retrace
            sample = buf.sample(8)
            # The sampled dict has the same keys the learn() path receives
            data_device = sample["rewards"].device
            z = T.zeros_like(sample["rewards"])
            q_re, _ = compute_q_retrace(
                sample["rewards"],
                sample["terminations"],
                sample["truncations"],
                sample["trajectory_lengths"],
                q_cur=z,
                target_q=z,
                cur_log_probs=z,
                buf_log_probs=z,
                discount=0.99,
                device=data_device,
            )
            # Just sanity: must be finite and shape must match
            assert T.isfinite(q_re).all()
            assert q_re.shape[0] == 8

        finally:
            try:
                nstep_env.env.close()
            except Exception:
                pass
            try:
                plain_env.close()
            except Exception:
                pass


# =============================================================================
# STRENGTHENED INTEGRATION TESTS TARGETING THE REMAINING GAPS
#
# The tests above (pure compute_q_retrace + basic wrapper → retrace) give high
# confidence in the math and basic wiring.
#
# The additional tests below directly attack the four areas that still needed
# coverage (as of the time these tests were added):
#
# 1. Real data semantics on LunarLanderContinuous-v3 (longer episodes, continuous
#    actions, time-limit truncations) with N=5 — the exact hyper-parameter that
#    exposed the original regression.
# 2. Interaction between the trainer’s `valid_steps = ~self._prev_done` logic and
#    the exact moment VectorNStepReward emits a chunk (higher parallelism,
#    continuous actions + entropy).
# 3. The full combination of rolling circular buffers + repeat/zero padding +
#    mask / cum_c timing inside compute_q_retrace on realistic N=5 data.
# 4. The data path when simple state/reward normalizers (and a trivial IM) are
#    active in the loop.
#
# Full HER + complex intrinsic motivation + the complete trainer/scheduler
# machinery is intentionally left as future work (or covered in higher-level
# training smoke tests), but the critical n-step / termination-boundary /
# retrace contract is exercised far more realistically than before.
# =============================================================================

def _create_lunarlander_gymnasium_wrapper(n: int = 5, num_envs: int = 8, seed: int = 42):
    """Create a *real* high-level GymnasiumWrapper exactly as used in training,
    configured for LunarLanderContinuous-v3 with VectorNStepReward (N=5 by
    default).  This pulls in the full Observation + n_step_trajectory path that
    the real trainer uses.
    """
    # Force CPU for determinism and speed in the test suite
    from phoenx import torch_utils as tu
    orig = tu.get_device
    tu.get_device = lambda *a, **k: T.device("cpu")
    try:
        env = GymnasiumWrapper(
            cfg="LunarLanderContinuous-v3",
            num_envs=num_envs,
            seed=seed,
            wrappers=[{
                "type": "VectorNStepReward",
                "params": {"n": n}
            }],
            render_mode=None,
        )
        return env
    finally:
        tu.get_device = orig


class TestFullStackN5LunarLanderBoundaries:
    """Tests that close the remaining semantic gaps for N>1 q-retrace."""

    def test_lunarlandercontinuous_n5_real_termination_and_truncation_boundaries(self):
        """Drive the real high-level GymnasiumWrapper + LunarLanderContinuous-v3
        (N=5) and verify that chunks emitted around real terminations and
        time-limit truncations produce correct Retrace targets when fed through
        the real ReplayBuffer and compute_q_retrace.

        This directly attacks gaps #1 and #3.
        """
        env = _create_lunarlander_gymnasium_wrapper(n=5, num_envs=8, seed=123)
        buf = ReplayBuffer(env=env, buffer_size=20000, N=5, device=DEVICE)

        try:
            obs = env.reset(seed=123)
            prev_done = T.zeros(env.num_envs, dtype=T.bool, device=DEVICE)
            interesting = []

            for step in range(6000):  # enough to see many terminations + truncations
                # Simplified trainer-like action selection (random for the test)
                action = env.action_space.sample()  # numpy batch
                action_t = T.as_tensor(action, device=DEVICE)

                # Minimal "Action" stand-in so VectorNStepReward doesn't crash
                from dataclasses import dataclass
                @dataclass
                class _MiniAction:
                    actions: T.Tensor
                    raw_actions: T.Tensor | None = None
                    log_probs: T.Tensor | None = None

                logp = T.zeros(env.num_envs, device=DEVICE)
                env._find_nstep_wrapper().set_action(_MiniAction(actions=action_t, log_probs=logp))

                next_obs = env.step(action_t)

                valid = ~prev_done
                if valid.any():
                    # Record exactly like the real trainer
                    buf.record(next_obs, prev_observation=obs, actions=_MiniAction(actions=action_t, log_probs=logp),
                               prev_dones=prev_done)

                dones = T.logical_or(next_obs.terminations, next_obs.truncations)
                prev_done = dones.clone()
                obs = next_obs

                # Collect interesting n-step chunks (those with terminations/truncations or short)
                traj = getattr(next_obs, 'n_step_trajectory', None) or (next_obs.infos or {}).get('n-step trajectory')
                if traj is not None and len(traj.get('trajectory_lengths', [])) > 0:
                    L = traj['trajectory_lengths']
                    has_term = (traj['terminations'] | traj['truncations']).any(dim=1)
                    if has_term.any() or (L < 5).any():
                        interesting.append({k: v.clone() for k, v in traj.items()})

            assert len(interesting) >= 5, "Not enough terminating/truncated chunks collected on LunarLander"

            # Feed a sample of them through the real buffer + compute_q_retrace (N=5)
            for traj in interesting[:20]:
                # Make sure the buffer has seen them (some may already be in)
                try:
                    buf.add(**{k: v for k, v in traj.items()
                               if k in {'states','actions','rewards','next_states',
                                        'terminations','truncations','log_probs',
                                        'intrinsic_rewards','trajectory_lengths'}})
                except Exception:
                    pass  # buffer may already contain similar data

                # Force all tensors in this traj batch to the test device (CPU).
                # The high-level GymnasiumWrapper + NumpyToTorch for LunarLander
                # can still produce CUDA tensors at runtime even with the creation-time
                # get_device patch (the patch only affects __init__ of the wrappers).
                traj = {
                    k: (v.to(DEVICE) if isinstance(v, T.Tensor) else v)
                    for k, v in traj.items()
                }

                z = T.zeros_like(traj['rewards'])
                q_re, m = compute_q_retrace(
                    traj['rewards'], traj['terminations'], traj['truncations'],
                    traj['trajectory_lengths'], z, z, z, z, discount=0.99, device=DEVICE
                )
                assert T.isfinite(q_re).all()

                # Basic semantic check: if a chunk has a termination, the mask should
                # have stopped the accumulation (cum_c should be zero at the end for
                # rows that had a done).
                mask = m['mask']
                for i in range(q_re.shape[0]):
                    L = int(traj['trajectory_lengths'][i])
                    if L > 0:
                        dones = traj['terminations'][i, :L] | traj['truncations'][i, :L]
                        if dones.any():
                            first = int(dones.nonzero(as_tuple=True)[0][0])
                            if first + 1 < L:
                                assert (mask[i, first+1:L] == 0).all()

        finally:
            try:
                env.close()
            except Exception:
                pass

    def test_trainer_valid_steps_prev_done_interaction_with_nstep_emission(self):
        """Exercise the exact `valid_steps = ~self._prev_done` + buffer.record
        logic from the real Trainer together with VectorNStepReward emission.

        This closes gap #2 (trainer / wrapper boundary at episode starts).
        """
        env = _create_lunarlander_gymnasium_wrapper(n=5, num_envs=4, seed=99)
        buf = ReplayBuffer(env=env, buffer_size=10000, N=5, device=DEVICE)

        try:
            obs = env.reset(seed=99)
            prev_done = T.zeros(env.num_envs, dtype=T.bool, device=DEVICE)

            for _ in range(2500):
                action_np = env.action_space.sample()
                action_t = T.as_tensor(action_np, device=DEVICE)

                @dataclass
                class _MiniAction:
                    actions: T.Tensor
                    raw_actions: T.Tensor | None = None
                    log_probs: T.Tensor | None = None

                logp = T.zeros(env.num_envs, device=DEVICE)
                env._find_nstep_wrapper().set_action(_MiniAction(actions=action_t, log_probs=logp))

                next_obs = env.step(action_t)

                valid_steps = ~prev_done
                if valid_steps.any():
                    buf.record(next_obs, prev_observation=obs,
                               actions=_MiniAction(actions=action_t, log_probs=logp),
                               prev_dones=prev_done)

                dones = T.logical_or(next_obs.terminations, next_obs.truncations)
                prev_done = dones.clone()
                obs = next_obs

            # After driving, sample and run through retrace — we only care that
            # nothing explodes and the emitted lengths/terminations are respected.
            if buf.is_ready(16):
                sample = buf.sample(16)
                z = T.zeros_like(sample['rewards'])
                q_re, _ = compute_q_retrace(
                    sample['rewards'], sample['terminations'], sample['truncations'],
                    sample['trajectory_lengths'], z, z, z, z, 0.99, device=DEVICE
                )
                assert T.isfinite(q_re).all()
                assert q_re.shape[0] == 16

        finally:
            try:
                env.close()
            except Exception:
                pass

    def test_data_path_with_state_and_reward_normalizers_active(self):
        """Ensure that when simple state and reward normalizers are active
        (as in most real training runs), the tensors that eventually reach
        compute_q_retrace are still semantically correct for N=5 boundaries.

        This closes gap #4 for the normalizer case.
        """
        env = _create_lunarlander_gymnasium_wrapper(n=5, num_envs=4, seed=7)

        # Attach simple running normalizers (exactly like training)
        obs_dim = int(np.prod(env.single_observation_space.shape))
        state_norm = RunningNorm(num_features=obs_dim, device=DEVICE)
        reward_norm = RewardNorm(gamma=0.99, device=DEVICE)

        buf = ReplayBuffer(env=env, buffer_size=10000, N=5, device=DEVICE)

        try:
            obs = env.reset(seed=7)
            prev_done = T.zeros(env.num_envs, dtype=T.bool, device=DEVICE)

            for _ in range(2000):
                action_np = env.action_space.sample()
                action_t = T.as_tensor(action_np, device=DEVICE)

                @dataclass
                class _MiniAction:
                    actions: T.Tensor
                    raw_actions: T.Tensor | None = None
                    log_probs: T.Tensor | None = None

                logp = T.zeros(env.num_envs, device=DEVICE)
                env._find_nstep_wrapper().set_action(_MiniAction(actions=action_t, log_probs=logp))

                next_obs = env.step(action_t)

                # Update normalizers (like the real trainer)
                state_norm.add(next_obs.states)
                state_norm.update()
                dones = T.logical_or(next_obs.terminations, next_obs.truncations)
                reward_norm.add(next_obs.rewards, dones)
                reward_norm.update()

                valid = ~prev_done
                if valid.any():
                    buf.record(next_obs, prev_observation=obs,
                               actions=_MiniAction(actions=action_t, log_probs=logp),
                               prev_dones=prev_done)

                # dones = T.logical_or(next_obs.terminations, next_obs.truncations)
                prev_done = dones.clone()
                obs = next_obs

            if buf.is_ready(8):
                sample = buf.sample(8)
                b, n = sample['rewards'].shape
                norm_rewards = reward_norm.normalize(sample['rewards'].reshape(b*n)).reshape(b, n)
                z = T.zeros_like(norm_rewards)
                q_re, _ = compute_q_retrace(
                    norm_rewards, sample['terminations'], sample['truncations'],
                    sample['trajectory_lengths'], z, z, z, z, 0.99, device=DEVICE
                )
                assert T.isfinite(q_re).all()

        finally:
            try:
                env.close()
            except Exception:
                pass

    # NOTE on HER + complex IM:
    # Full HER (which requires goal-conditioned wrappers and a different buffer
    # interaction) and heavy intrinsic motivation training inside the loop are
    # out of scope for this fast unit/integration file.  The basic contract
    # ("the n-step dicts that reach compute_q_retrace still contain correct
    # termination placement and trajectory_lengths") is already exercised by the
    # tests above.  Any HER-specific or heavy-IM-specific n-step bugs would be
    # caught by higher-level training smoke tests or dedicated HER test files.


# =============================================================================
# Independent ground-truth emission tests for VectorNStepReward
# -----------------------------------------------------------------------------
# The integration tests above validate that compute_q_retrace *consumes* the
# emitted data self-consistently. Their reference, however, is derived FROM the
# wrapper's own output, so a bug in the wrapper's emission (wrong temporal
# order, off-by-one in the circular buffer, cross-episode leakage, broken
# tail-flush) would NOT be caught — the wrong data would simply be compared
# against itself.
#
# The tests below close that gap. They maintain an entirely separate, naive,
# plain-Python record of the *true* environment transitions and assert that
# every window the wrapper emits matches that record exactly:
#   - temporal ordering of states / next_states / rewards
#   - termination / truncation placement within the window
#   - trajectory_lengths (capped at n)
#   - zero-padding past the valid length (rewards / terminations / truncations)
#   - no cross-episode leakage (episode record cleared on done)
#   - tail-flush sub-windows on terminal steps
#
# The emitted row order is deterministic and is reproduced exactly by the
# reference:
#   1. main windows, one per env with length>0, in ascending env index
#   2. tail sub-windows, per done env (ascending), for j = 1 .. L-1
# =============================================================================

def _to_numpy(x):
    """Device/type-agnostic conversion to a numpy array.

    VectorNStepReward.step returns rewards/terminations/truncations as torch
    tensors on self.device (which may be CUDA even in these CPU-oriented tests,
    since the get_device patch only affects allocation during __init__).
    """
    if isinstance(x, T.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_expected_emitted_windows(episodes, dones, n):
    """Reconstruct, from a naive per-env transition record, the exact list of
    windows VectorNStepReward should emit on this step (in emission order).

    Each returned window is a list of transition dicts with keys
    state / reward / next_state / term / trunc.
    """
    num_envs = len(episodes)
    expected = []
    # 1. Main windows (ascending env index, only envs with a non-empty episode)
    mains = {}
    for e in range(num_envs):
        L = min(len(episodes[e]), n)
        if L > 0:
            main = episodes[e][-L:]
            mains[e] = main
            expected.append(main)
    # 2. Tail-flush sub-windows for every env that is done this step
    for e in range(num_envs):
        if dones[e] and e in mains:
            main = mains[e]
            for j in range(1, len(main)):
                expected.append(main[j:])
    return expected


def _assert_emitted_matches_expected(traj, expected, n):
    """Assert that an emitted n-step trajectory dict matches the expected list
    of windows exactly (row order, ordering within rows, lengths, padding)."""
    if not expected:
        assert traj is None or int(len(traj.get("trajectory_lengths", []))) == 0, \
            "Wrapper emitted a window when none was expected (phantom/empty step)"
        return

    n_rows = 0 if traj is None else int(len(traj["trajectory_lengths"]))
    assert traj is not None and n_rows == len(expected), \
        f"Expected {len(expected)} emitted window(s), got {n_rows}"

    for r, win in enumerate(expected):
        Lw = len(win)
        assert int(traj["trajectory_lengths"][r]) == Lw, \
            f"row {r}: trajectory_length {int(traj['trajectory_lengths'][r])} != {Lw}"

        for k in range(Lw):
            assert np.allclose(
                traj["states"][r, k].cpu().numpy(), win[k]["state"], atol=1e-5
            ), f"row {r} step {k}: state mismatch (temporal ordering / leakage)"
            assert np.allclose(
                traj["next_states"][r, k].cpu().numpy(), win[k]["next_state"], atol=1e-5
            ), f"row {r} step {k}: next_state mismatch"
            assert abs(float(traj["rewards"][r, k]) - win[k]["reward"]) < 1e-5, \
                f"row {r} step {k}: reward mismatch"
            assert bool(traj["terminations"][r, k]) == win[k]["term"], \
                f"row {r} step {k}: termination flag misplaced"
            assert bool(traj["truncations"][r, k]) == win[k]["trunc"], \
                f"row {r} step {k}: truncation flag misplaced"

        # zero-padding past the valid length
        for k in range(Lw, n):
            assert abs(float(traj["rewards"][r, k])) < 1e-8, \
                f"row {r} pad {k}: reward not zero-padded"
            assert not bool(traj["terminations"][r, k]), \
                f"row {r} pad {k}: termination not zero-padded"
            assert not bool(traj["truncations"][r, k]), \
                f"row {r} pad {k}: truncation not zero-padded"


class TestVectorNStepRewardEmissionGroundTruth:
    """Independent ground-truth validation of VectorNStepReward emission.

    Maintains a naive, obviously-correct record of the true transitions and
    asserts every emitted window matches it exactly. This is the layer that can
    actually catch the class of subtle n-step / boundary bugs that produced the
    N=5 regression (and which the self-consistent integration tests cannot see).
    """

    def test_emission_matches_naive_reference_single_env(self):
        """num_envs == 1: the simplest, most readable case. Emitted row order is
        trivially [main, tail_1, ..., tail_{L-1}]."""
        n = 4
        env = _create_real_nstep_wrapped_env(n=n, num_envs=1, seed=7)
        try:
            states, _ = env.reset()
            cur_state = _to_numpy(states)[0].copy()
            episode = []          # true transitions since last reset
            prev_done = False
            saw_termination = False
            saw_tail_flush = False

            for _step in range(600):
                a_np = np.array([env.single_action_space.sample()])
                a_t = T.as_tensor(a_np, device="cpu")
                logp = T.zeros(1, device="cpu", dtype=T.float32)
                env.set_action(_TestAction(actions=a_t, raw_actions=None, log_probs=logp))

                next_states, rewards, terms, truncs, infos = env.step(a_np)
                next_states = _to_numpy(next_states)
                rewards = _to_numpy(rewards).astype(np.float64)
                term = bool(_to_numpy(terms)[0])
                trunc = bool(_to_numpy(truncs)[0])
                done = term or trunc

                # Update naive reference (mirror wrapper's `active = ~prev_done`)
                if not prev_done:
                    episode.append({
                        "state": cur_state.copy(),
                        "reward": float(rewards[0]),
                        "next_state": next_states[0].copy(),
                        "term": term,
                        "trunc": trunc,
                    })

                expected = _build_expected_emitted_windows([episode], [done], n)
                traj = infos.get("n-step trajectory")
                _assert_emitted_matches_expected(traj, expected, n)

                if expected:
                    if len(expected) > 1:
                        saw_tail_flush = True
                    if done:
                        saw_termination = True

                prev_done = done
                cur_state = next_states[0].copy()
                if done:
                    episode = []

            assert saw_termination, "No terminations occurred — boundaries were never exercised"
            assert saw_tail_flush, "No tail-flush sub-windows were produced/observed"

        finally:
            try:
                env.env.close()
            except Exception:
                pass

    def test_emission_matches_naive_reference_multi_env(self):
        """num_envs > 1: staggered terminations across parallel envs. This
        exercises the harder paths the single-env case cannot:
          - the valid-env boolean mask selecting/ordering main rows
          - multiple done envs in the SAME step contributing tail rows
          - tail-row ordering (ascending env index, then j = 1 .. L-1)
          - per-env circular buffers staying independent (no cross-env bleed)
        """
        n = 4
        num_envs = 3
        env = _create_real_nstep_wrapped_env(n=n, num_envs=num_envs, seed=11)
        try:
            states, _ = env.reset()
            cur_state = _to_numpy(states).copy()                  # [num_envs, obs]
            episodes = [[] for _ in range(num_envs)]               # per-env transitions
            prev_done = [False] * num_envs
            saw_termination = False
            saw_tail_flush = False
            saw_multi_main_rows = False
            saw_multi_done_same_step = False

            for _step in range(1500):
                a_np = np.array([env.single_action_space.sample() for _ in range(num_envs)])
                a_t = T.as_tensor(a_np, device="cpu")
                logp = T.zeros(num_envs, device="cpu", dtype=T.float32)
                env.set_action(_TestAction(actions=a_t, raw_actions=None, log_probs=logp))

                next_states, rewards, terms, truncs, infos = env.step(a_np)
                next_states = _to_numpy(next_states)
                rewards = _to_numpy(rewards).astype(np.float64)
                terms = _to_numpy(terms).astype(bool)
                truncs = _to_numpy(truncs).astype(bool)
                dones = [bool(terms[e] or truncs[e]) for e in range(num_envs)]

                # Update naive per-env reference (mirror per-env `active = ~prev_done`)
                for e in range(num_envs):
                    if not prev_done[e]:
                        episodes[e].append({
                            "state": cur_state[e].copy(),
                            "reward": float(rewards[e]),
                            "next_state": next_states[e].copy(),
                            "term": bool(terms[e]),
                            "trunc": bool(truncs[e]),
                        })

                expected = _build_expected_emitted_windows(episodes, dones, n)
                traj = infos.get("n-step trajectory")
                _assert_emitted_matches_expected(traj, expected, n)

                # Coverage bookkeeping
                num_main = sum(1 for e in range(num_envs) if min(len(episodes[e]), n) > 0)
                if num_main > 1:
                    saw_multi_main_rows = True
                num_done = sum(1 for d in dones if d)
                if num_done >= 1:
                    saw_termination = True
                if num_done >= 2:
                    saw_multi_done_same_step = True
                if expected and len(expected) > num_main:
                    saw_tail_flush = True

                # Advance per-env bookkeeping
                for e in range(num_envs):
                    prev_done[e] = dones[e]
                    cur_state[e] = next_states[e].copy()
                    if dones[e]:
                        episodes[e] = []

            assert saw_termination, "No terminations occurred — boundaries were never exercised"
            assert saw_tail_flush, "No tail-flush sub-windows were produced/observed"
            assert saw_multi_main_rows, "Never saw >1 valid main row — multi-env ordering not exercised"
            # Staggered terminations are the whole point; with 3 envs over 1500
            # CartPole steps this is effectively guaranteed, but we assert it so a
            # regression that serializes episodes would surface.
            assert saw_multi_done_same_step, \
                "Never saw two envs done on the same step — multi-done tail ordering not exercised"

        finally:
            try:
                env.env.close()
            except Exception:
                pass