"""Smoke training gates: short seeded runs that prove the library still learns.

Run with ``pytest -m smoke``. See ``.cursor/skills/behavior-gates/SKILL.md``.

These are the only tests in the suite that would notice a silent numerical
defect. A sign error in an advantage, a bootstrap across a truncation boundary,
or a normalizer that fails to restore does not raise and does not fail a shape
assertion — it produces a learning curve that is merely worse. Nothing else
here checks for that.

RULES
-----
1. Never lower a threshold to make a red gate green. Bisect instead.
2. Thresholds sit at roughly half of what the config reliably reaches, so
   ordinary variance never trips them and a real regression always does.
3. Two seeds minimum, asserting on the worse of the two.
4. Under ~90 seconds per test.
5. Go through the same public entry point users go through (``load_config`` +
   ``build_trainer_from_config``, the pair ``phoenx-train`` uses), so config
   schema drift is gated here too.

WIRING
------
``train_and_evaluate`` trains from a fixture config through the public
train/eval path and returns the mean evaluation return. The three learning
tests (``test_ppo_learns_cartpole``, ``test_sac_learns_pendulum``,
``test_td3_learns_pendulum``) are skipped until thresholds measured on seeds 0
and 1 are approved (Phase 4b); ``test_training_is_seed_deterministic`` is not
skipped and must pass today.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as T

from phoenx.builder import build_trainer_from_config, load_config
from phoenx.trainer import Trainer

pytestmark = [pytest.mark.smoke, pytest.mark.slow]

FIXTURES = Path(__file__).parent / "fixtures"


def train_and_evaluate(config_path, seed, total_steps, eval_episodes=10):
    """Train an agent from a config and return its mean evaluation return.

    Goes through the same public entry point ``phoenx-train`` uses
    (``load_config`` + ``build_trainer_from_config``), so config schema drift
    is gate-kept here too. ``schedule.seed`` and ``env.config.seed`` are both
    overridden to ``seed`` so training and the env RNG start from the same
    state; ``schedule.stop_units`` is overridden to ``total_steps`` and
    ``save_dir`` to a fresh temp directory per call.

    Args:
        config_path (pathlib.Path): Path to the fixture config (``.yml``).
        seed (int): Seed applied to the agent, the environment, and (via the
            reloaded trainer's own config) the evaluation environment.
        total_steps (int): Environment-step budget for training.
        eval_episodes (int): Deterministic evaluation episodes to average.

    Returns:
        float: Mean evaluation return (``episode_reward``) over
            ``eval_episodes``, i.e. ``sum(_score_history) / len(_score_history)``.
    """
    config = load_config(config_path)
    config["schedule"]["seed"] = seed
    config["env"]["config"]["seed"] = seed
    config["schedule"]["stop_units"] = total_steps
    config["log_level"] = "ERROR"

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = str(Path(tmp_dir) / "run")
        config["save_dir"] = save_dir

        # phoenx.env_wrapper's tensor conversion (GymnasiumWrapper._initialize_env,
        # extract_states_goals, VectorNStepReward) calls get_device() with no
        # argument, which always resolves to CUDA when a GPU is present -- it does
        # not read env.config or agent.config.device. On a CUDA-capable machine
        # this crashes a "device: cpu" fixture with a device-mismatch RuntimeError
        # the moment the agent (cpu) and env (cuda) tensors meet in Trainer.step.
        # There is no config knob for this in src/phoenx (out of scope here), so
        # this test-only patch pins env_wrapper's device resolution to cpu for the
        # duration of the run, matching what the fixture's device: cpu declares.
        with patch("phoenx.env_wrapper.get_device", return_value=T.device("cpu")):
            trainer = build_trainer_from_config(config)
            trainer.train()  # closes env
            trainer.save()  # AFTER train, so the checkpoint is the final weights

            loaded = Trainer.load(save_dir)
            try:
                loaded.test(unit="episode", units=eval_episodes)
                return sum(loaded._score_history) / len(loaded._score_history)
            finally:
                loaded.env.close()


def _assert_learns(name, config, seeds, total_steps, threshold):
    """Assert the worst seed clears the threshold.

    Args:
        name (str): Test identifier (used only in the failure message).
        config (pathlib.Path): Config to train from.
        seeds (list[int]): Seeds to run.
        total_steps (int): Environment-step budget per seed.
        threshold (float): Minimum acceptable mean evaluation return.
    """
    results = {}
    for seed in seeds:
        achieved = train_and_evaluate(config, seed=seed, total_steps=total_steps)
        results[seed] = achieved

    worst_seed = min(results, key=results.get)
    assert results[worst_seed] >= threshold, (
        f"{name}: worst seed {worst_seed} reached {results[worst_seed]:.1f}, "
        f"threshold {threshold}. All seeds: {results}. "
        "Do NOT lower this threshold — bisect with `git bisect run pytest -m smoke`."
    )


# --------------------------------------------------------------------------
# One test per algorithm family. Add HER once a fast goal-conditioned env exists.
#
# KNOWN ISSUE (found while measuring, not fixed here -- src/phoenx/** is out of
# scope for this change): ReplayBuffer.sample (src/phoenx/buffer.py:444,
# `self.gen = np.random.default_rng()`) draws OS entropy instead of reading the
# seed set_seed/schedule.seed establishes. Every SAC/TD3 (off-policy) run above
# therefore samples different minibatches across otherwise-identical seeded
# runs, so the seed 0 / seed 1 numbers measured for those two tests are single
# noisy draws, not reproducible baselines -- re-running the same seed can move
# the achieved return by 1000+ points. PPO's RolloutBuffer.sample returns the
# whole buffer (no RNG draw), which is why the PPO fixture is fully
# reproducible and test_training_is_seed_deterministic passes on it below. A
# smaller, separate latent bug: Trainer._initialize_run
# (src/phoenx/trainer.py:340) does `self.schedule.seed if self.schedule.seed
# else <draw a new seed>`, so schedule.seed == 0 is falsy and silently
# replaced -- harmless for the fully-deterministic PPO path (the replacement
# draw is itself deterministic there) but worth knowing before trusting a
# `seed: 0` fixture value at face value. Do not fix either in src/phoenx here;
# report and let the user decide.
# --------------------------------------------------------------------------


@pytest.mark.skip(reason="threshold not approved")
def test_ppo_learns_cartpole():
    """On-policy path still learns: PPO on CartPole clears half of solved."""
    _assert_learns(
        name="ppo_cartpole",
        config=FIXTURES / "smoke_ppo_cartpole.yml",
        seeds=[0, 1],
        total_steps=60_000,
        threshold=120.0,  # solved is 500; set after measuring, keep well below
    )


@pytest.mark.skip(reason="threshold not approved")
def test_sac_learns_pendulum():
    """Off-policy path still learns: SAC on Pendulum clears a weak threshold."""
    _assert_learns(
        name="sac_pendulum",
        config=FIXTURES / "smoke_sac_pendulum.yml",
        seeds=[0, 1],
        total_steps=6_000,  # shrunk from the 20_000 draft: measured wall time
        # exceeded 90s per run at 15_000-20_000 steps on this CPU.
        threshold=-800.0,  # random is about -1200, trained about -200
    )


@pytest.mark.skip(reason="threshold not approved")
def test_td3_learns_pendulum():
    """Deterministic off-policy path still learns, with its own target-smoothing path."""
    _assert_learns(
        name="td3_pendulum",
        config=FIXTURES / "smoke_td3_pendulum.yml",
        seeds=[0, 1],
        total_steps=15_000,  # shrunk from the 20_000 draft; ~63s per run measured
        threshold=-800.0,
    )


def test_training_is_seed_deterministic():
    """The same seed twice produces the same evaluation return.

    A flaky smoke gate is worse than no gate, and non-determinism here means
    every other threshold in this file is measuring noise as much as skill.
    """
    config = FIXTURES / "smoke_ppo_cartpole.yml"
    first = train_and_evaluate(config, seed=7, total_steps=10_000)
    second = train_and_evaluate(config, seed=7, total_steps=10_000)
    assert first == pytest.approx(second, rel=1e-6), (
        f"same seed produced {first} then {second}; seeding is incomplete "
        "(check env, action sampling, network init, and buffer sampling)"
    )
