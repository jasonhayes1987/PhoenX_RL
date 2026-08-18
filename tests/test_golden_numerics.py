"""Golden numerical tests: PhoenX against independently written references.

Run with ``pytest -m golden``. See ``.cursor/skills/run-tests/SKILL.md``.

The rule that makes this file worth anything: **every expected value here is
produced by code written independently of PhoenX** — a closed-form identity or
a slow, obvious reference implementation defined below. Never rewrite one of
these tests to compute its expectation by calling the production code path. A
test that asserts the code equals itself passes on every bug it contains.

Independent reference-vs-reference identities may use a tight tolerance
(float64, ``atol=1e-9``). PhoenX runs in float32, so PhoenX-vs-reference
comparisons use ``rtol=0, atol=1e-4``. A gap larger than that is a real
disagreement, not something to paper over by loosening the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as T

pytestmark = pytest.mark.golden

# --------------------------------------------------------------------------
# ADAPTER — wires this file's calling convention onto the real PhoenX API.
# --------------------------------------------------------------------------

try:
    from phoenx.agent_utils import compute_advantages_and_returns as phoenx_gae
except Exception:  # pragma: no cover - adapter not wired yet
    phoenx_gae = None

try:
    from phoenx.normalizer import RunningNorm
except Exception:  # pragma: no cover - adapter not wired yet
    RunningNorm = None


def call_phoenx_gae(
    rewards, values, next_value, terminated, truncated, gamma, lam, bootstrap_truncations=True
):
    """Adapt PhoenX's GAE signature to this file's calling convention.

    PhoenX does not expose a ``compute_gae(rewards, values, ...)`` entry
    point; production computes TD errors first and feeds those into
    ``compute_gae``. This helper drives that same path through
    ``compute_advantages_and_returns`` so the test still speaks in terms of
    rewards/values/next_value like the independent reference below.

    ``next_values[t] = values[t + 1]`` for ``t < T - 1``, and the final step's
    next value is the scalar ``next_value`` bootstrap — this is a convention
    for this test's 1-D fixtures, not a claim about how vectorized
    autoreset environments populate next-states in production.

    Args:
        rewards (np.ndarray): Rewards, shape (T,).
        values (np.ndarray): Value estimates for the visited states, shape (T,).
        next_value (float): Bootstrap value for the state after the last step.
        terminated (np.ndarray): Boolean MDP terminations, shape (T,).
        truncated (np.ndarray): Boolean time-limit truncations, shape (T,).
        gamma (float): Discount factor.
        lam (float): GAE lambda.
        bootstrap_truncations (bool): Forwarded to
            ``compute_advantages_and_returns``; ``True`` keeps ``gamma * V(s')``
            on truncated steps, ``False`` drops it on both terminations and
            truncations.

    Returns:
        np.ndarray: Advantages, shape (T,).
    """
    timesteps = len(rewards)
    rewards_t = T.as_tensor(np.asarray(rewards), dtype=T.float32).reshape(timesteps, 1)
    values_t = T.as_tensor(np.asarray(values), dtype=T.float32).reshape(timesteps, 1)
    next_values_np = np.concatenate([np.asarray(values)[1:], [next_value]])
    next_values_t = T.as_tensor(next_values_np, dtype=T.float32).reshape(timesteps, 1)
    terminations_t = T.as_tensor(np.asarray(terminated), dtype=T.bool).reshape(timesteps, 1)
    truncations_t = T.as_tensor(np.asarray(truncated), dtype=T.bool).reshape(timesteps, 1)

    advantages, _returns, _td_errors = phoenx_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminations_t,
        truncations_t,
        gamma,
        lam,
        bootstrap_truncations,
        device="cpu",
    )
    return advantages.squeeze(-1).cpu().numpy()


class PhoenxNormalizer:
    """Thin adapter from this file's calling convention to ``RunningNorm``.

    ``RunningNorm`` lives in ``phoenx.normalizer`` and separates accumulation
    (``add``) from committing accumulated statistics into the running
    mean/var (``update``). This wrapper collapses that into a single
    ``update(chunk)`` call and exposes the running statistics under the
    ``mean`` / ``var`` / ``count`` names this file's tests use.

    Args:
        num_features (int): Feature dimension tracked by the wrapped
            ``RunningNorm``.

    Example:
        >>> norm = PhoenxNormalizer(4)
        >>> norm.update(np.random.randn(32, 4))
        >>> norm.mean.shape
        torch.Size([4])
    """

    def __init__(self, num_features):
        self._norm = RunningNorm(num_features, device="cpu")

    def update(self, chunk):
        """Accumulate ``chunk`` and immediately commit it to running stats.

        Args:
            chunk (np.ndarray): Batch of samples, shape (n, num_features).
        """
        self._norm.add(T.as_tensor(np.asarray(chunk), dtype=T.float32))
        self._norm.update()

    @property
    def mean(self):
        """torch.Tensor: Running mean, shape (num_features,)."""
        return self._norm.running_mean

    @property
    def var(self):
        """torch.Tensor: Running variance, shape (num_features,)."""
        return self._norm.running_var

    @property
    def count(self):
        """int: Total number of samples committed via ``update``."""
        return int(self._norm.running_cnt.item())

    def save(self, path):
        """Persist running statistics to ``path`` via ``RunningNorm.save_state``.

        Args:
            path (str | Path): Destination file path.
        """
        self._norm.save_state(path)

    def load(self, path):
        """Restore running statistics from ``path`` via ``RunningNorm.load_state``.

        Args:
            path (str | Path): Path previously written by ``save``.
        """
        self._norm.load_state(path)

    def normalize(self, x):
        """Forward to ``RunningNorm.normalize``; accepts a numpy array.

        Args:
            x (np.ndarray): Input batch, trailing dim ``num_features``.

        Returns:
            torch.Tensor: Normalized batch (callers here only check that
                running statistics did not move, so the return value itself
                is not asserted on).
        """
        return self._norm.normalize(T.as_tensor(np.asarray(x), dtype=T.float32))

    def train(self):
        """Put the wrapped ``RunningNorm`` in train mode.

        Returns:
            PhoenxNormalizer: This instance, for chaining.
        """
        self._norm.train()
        return self

    def eval(self):
        """Put the wrapped ``RunningNorm`` in eval mode.

        Returns:
            PhoenxNormalizer: This instance, for chaining.
        """
        self._norm.eval()
        return self


if RunningNorm is None:  # pragma: no cover - adapter not wired yet
    PhoenxNormalizer = None


# --------------------------------------------------------------------------
# Reference implementations — independent of PhoenX. Do not "fix" these.
# --------------------------------------------------------------------------


def reference_gae(rewards, values, next_value, terminated, truncated, gamma, lam):
    """Compute GAE with an explicit, deliberately slow backward loop.

    Bootstrapping uses ``terminated`` only: a time-limit truncation is not an
    MDP terminal, so its successor value is still bootstrapped. Accumulation of
    the GAE trace resets at either kind of episode boundary.

    Args:
        rewards (np.ndarray): Rewards, shape (T,).
        values (np.ndarray): Value estimates, shape (T,).
        next_value (float): Bootstrap value after the final step.
        terminated (np.ndarray): Boolean MDP terminations, shape (T,).
        truncated (np.ndarray): Boolean time-limit truncations, shape (T,).
        gamma (float): Discount factor.
        lam (float): GAE lambda.

    Returns:
        np.ndarray: Advantages, shape (T,).
    """
    T_len = len(rewards)
    adv = np.zeros(T_len, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T_len)):
        v_next = next_value if t == T_len - 1 else values[t + 1]
        nonterminal = 0.0 if terminated[t] else 1.0
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        boundary = terminated[t] or truncated[t]
        running = delta + gamma * lam * (0.0 if boundary else 1.0) * running
        adv[t] = running
    return adv


def reference_returns_to_go(rewards, values, next_value, terminated, truncated, gamma):
    """Compute discounted return-to-go with bootstrapping at episode ends.

    Args:
        rewards (np.ndarray): Rewards, shape (T,).
        values (np.ndarray): Value estimates, shape (T,).
        next_value (float): Bootstrap value after the final step.
        terminated (np.ndarray): Boolean MDP terminations, shape (T,).
        truncated (np.ndarray): Boolean time-limit truncations, shape (T,).
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Returns, shape (T,).
    """
    T_len = len(rewards)
    out = np.zeros(T_len, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T_len)):
        v_next = next_value if t == T_len - 1 else values[t + 1]
        if terminated[t]:
            running = rewards[t]
        elif truncated[t] or t == T_len - 1:
            # Truncation, and the end of the rollout buffer, both bootstrap:
            # the future exists, it is simply not in this batch.
            running = rewards[t] + gamma * v_next
        else:
            running = rewards[t] + gamma * running
        out[t] = running
    return out


def make_rollout(seed, num_steps=64, term_p=0.05, trunc_p=0.03):
    """Generate a random rollout with occasional terminations and truncations.

    Args:
        seed (int): RNG seed.
        num_steps (int): Number of timesteps.
        term_p (float): Per-step probability of an MDP termination.
        trunc_p (float): Per-step probability of a time-limit truncation.

    Returns:
        tuple: rewards, values, next_value, terminated, truncated.
    """
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=num_steps)
    values = rng.normal(size=num_steps)
    terminated = rng.random(num_steps) < term_p
    truncated = (rng.random(num_steps) < trunc_p) & ~terminated
    return rewards, values, float(rng.normal()), terminated, truncated


# --------------------------------------------------------------------------
# Identity tests — these hold regardless of implementation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_reference_gae_lambda_one_equals_returns_minus_values(seed):
    """GAE at lambda=1 equals return-to-go minus the value baseline."""
    r, v, nv, term, trunc = make_rollout(seed)
    adv = reference_gae(r, v, nv, term, trunc, gamma=0.99, lam=1.0)
    ret = reference_returns_to_go(r, v, nv, term, trunc, gamma=0.99)
    np.testing.assert_allclose(adv, ret - v, rtol=0, atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_reference_gae_lambda_zero_equals_td_residual(seed):
    """GAE at lambda=0 equals the one-step TD residual."""
    r, v, nv, term, trunc = make_rollout(seed)
    adv = reference_gae(r, v, nv, term, trunc, gamma=0.99, lam=0.0)
    v_next = np.concatenate([v[1:], [nv]])
    expected = r + 0.99 * v_next * (~term) - v
    np.testing.assert_allclose(adv, expected, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------
# PhoenX vs reference.
# --------------------------------------------------------------------------


@pytest.mark.skipif(phoenx_gae is None, reason="GAE adapter not wired")
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("seed", range(3))
def test_phoenx_gae_matches_reference(lam, seed):
    """PhoenX's GAE matches an independent backward-loop reference."""
    r, v, nv, term, trunc = make_rollout(seed)
    got = call_phoenx_gae(r, v, nv, term, trunc, gamma=0.99, lam=lam)
    want = reference_gae(r, v, nv, term, trunc, gamma=0.99, lam=lam)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)


@pytest.mark.skipif(phoenx_gae is None, reason="GAE adapter not wired")
def test_truncation_bootstraps_and_termination_does_not():
    """A truncated step bootstraps its successor value; a terminated one does not.

    Uses the default ``bootstrap_truncations=True``: ``dones = terminations``
    only, so truncation keeps ``gamma * V(s')`` while termination drops it.
    """
    r = np.array([1.0, 1.0])
    v = np.array([0.5, 0.5])
    nv, gamma = 10.0, 0.99

    term_only = call_phoenx_gae(
        r, v, nv, np.array([False, True]), np.array([False, False]), gamma, 1.0
    )
    trunc_only = call_phoenx_gae(
        r, v, nv, np.array([False, False]), np.array([False, True]), gamma, 1.0
    )

    assert trunc_only[-1] > term_only[-1], (
        "truncation must bootstrap the successor value while termination must not; "
        f"got truncated={trunc_only[-1]!r} terminated={term_only[-1]!r}"
    )


@pytest.mark.skipif(phoenx_gae is None, reason="GAE adapter not wired")
def test_truncation_equals_termination_when_bootstrap_false():
    """With ``bootstrap_truncations=False``, truncation and termination agree.

    ``dones = terminations | truncations`` in this mode, so both a truncated
    and a terminated last step drop ``gamma * V(s')`` and their advantages
    must match within the PhoenX-vs-reference tolerance.
    """
    r = np.array([1.0, 1.0])
    v = np.array([0.5, 0.5])
    nv, gamma = 10.0, 0.99

    term_only = call_phoenx_gae(
        r, v, nv, np.array([False, True]), np.array([False, False]), gamma, 1.0,
        bootstrap_truncations=False,
    )
    trunc_only = call_phoenx_gae(
        r, v, nv, np.array([False, False]), np.array([False, True]), gamma, 1.0,
        bootstrap_truncations=False,
    )

    np.testing.assert_allclose(trunc_only[-1], term_only[-1], rtol=0, atol=1e-4)


@pytest.mark.skipif(PhoenxNormalizer is None, reason="Normalizer adapter not wired")
@pytest.mark.parametrize("seed", range(3))
def test_normalizer_incremental_matches_batch(seed):
    """Incremental updates over chunks equal a batch computation over their union."""
    rng = np.random.default_rng(seed)
    chunks = [rng.normal(loc=3.0, scale=2.0, size=(rng.integers(5, 50), 4)) for _ in range(6)]

    norm = PhoenxNormalizer(4)
    for chunk in chunks:
        norm.update(chunk)

    allx = np.concatenate(chunks, axis=0)
    np.testing.assert_allclose(np.asarray(norm.mean), allx.mean(axis=0), rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(norm.var), allx.var(axis=0), rtol=0, atol=1e-4)


@pytest.mark.skipif(PhoenxNormalizer is None, reason="Normalizer adapter not wired")
def test_normalizer_round_trips_through_checkpoint(tmp_path):
    """Save/restore preserves statistics exactly and the next update agrees.

    This is the regression test for the trial-boundary serialization defect in
    the hazard ledger. Identical statistics after a reload is necessary but not
    sufficient — the second half asserts that the RESTORED object continues to
    behave like the one that was never interrupted.
    """
    rng = np.random.default_rng(0)
    warmup = rng.normal(size=(200, 4))
    followup = rng.normal(size=(50, 4))

    live = PhoenxNormalizer(4)
    live.update(warmup)

    path = tmp_path / "norm.pt"
    live.save(path)
    restored = PhoenxNormalizer(4)
    restored.load(path)

    np.testing.assert_array_equal(np.asarray(restored.mean), np.asarray(live.mean))
    np.testing.assert_array_equal(np.asarray(restored.var), np.asarray(live.var))
    assert restored.count == live.count

    live.update(followup)
    restored.update(followup)
    np.testing.assert_allclose(
        np.asarray(restored.mean), np.asarray(live.mean), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(restored.var), np.asarray(live.var), rtol=0, atol=1e-4
    )


@pytest.mark.skipif(PhoenxNormalizer is None, reason="Normalizer adapter not wired")
@pytest.mark.parametrize("training", [True, False])
def test_normalizer_statistics_frozen_during_evaluation(training):
    """Neither train- nor eval-mode ``normalize`` calls may move running stats.

    ``RunningNorm.normalize`` always reads tracked ``running_mean`` /
    ``running_std`` regardless of ``training`` (decision 2 in the plan) — only
    ``add`` / ``update`` change them. This checks both modes so a future
    change that makes ``normalize`` train-mode-sensitive gets caught.
    """
    rng = np.random.default_rng(1)
    norm = PhoenxNormalizer(4)
    norm.update(rng.normal(size=(100, 4)))

    before = (np.array(norm.mean), np.array(norm.var), norm.count)
    if training:
        norm.train()
    else:
        norm.eval()
    for _ in range(20):
        norm.normalize(rng.normal(size=(1, 4)))

    np.testing.assert_array_equal(np.asarray(norm.mean), before[0])
    np.testing.assert_array_equal(np.asarray(norm.var), before[1])
    assert norm.count == before[2]
