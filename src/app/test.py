"""Evaluate / watch a trained PhoenX_RL agent act in its environment.

Loads a saved agent directory (e.g. a SAC run) and runs the policy so you can
watch it. Works for both Isaac Sim and Gymnasium agents: the environment is
rebuilt from the saved policy config, with ``render_mode`` and ``num_envs``
overridden for viewing.

- Isaac Sim opens its GUI window for any ``render_mode`` other than ``"headless"``.
- Gymnasium opens a window for ``render_mode "human"``.

The agent's saved observation/goal normalizers are loaded and applied (frozen),
exactly as during training, so the policy receives the inputs it expects.

Examples (run from the repo root, inside the ``rl_env`` conda env):

    python src/app/test.py --agent_dir "src/Trained_Models/IsaacSim/Franka/Reach/JointPos/SAC_3" --num_envs 1 --num_episodes 10
    python src/app/test.py --agent_dir "path/to/LunarLanderContinuous-v3/SAC_1" --render_mode human
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as T

from app.env_wrapper import EnvWrapper
from app.agent_utils import load_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test")

# Training-only wrappers that must not run during evaluation: they require a
# per-step ``set_action`` call and only build n-step trajectories for the buffer.
# They do not change the observation/action spaces, so dropping them is safe.
_TRAIN_ONLY_WRAPPERS = {"VectorNStepReward", "NStepReward"}


def build_eval_env(agent_dir: Path, num_envs, render_mode, seed) -> EnvWrapper:
    """Rebuild the env from the saved policy config, overridden for viewing."""
    with open(agent_dir / "policy" / "config.json", encoding="utf-8") as f:
        policy_cfg = json.load(f)
    spec = json.loads(policy_cfg["env"])  # {"type": ..., "config": {...}}
    env_cfg = dict(spec["config"])
    if num_envs is not None:
        env_cfg["num_envs"] = num_envs
    if render_mode is not None:
        env_cfg["render_mode"] = render_mode
    if seed is not None:
        env_cfg["seed"] = seed
    env_cfg["wrappers"] = [
        w for w in (env_cfg.get("wrappers") or [])
        if w.get("type") not in _TRAIN_ONLY_WRAPPERS
    ]
    spec["config"] = env_cfg
    return EnvWrapper.from_json(json.dumps(spec))


def select_action(agent, states, goals, deterministic: bool):
    """Deterministic = the policy's mean action (eval); else a sample from it."""
    with T.no_grad():
        if deterministic:
            return agent.act(states, goals, context="test").actions
        dist = agent.policy(states, goals)
        return dist.sample()


def run(agent, env: EnvWrapper, num_episodes: int, deterministic: bool, seed):
    """Roll out the policy, printing per-episode reward. Episode accounting mirrors
    Trainer.step (the ``~prev_done`` mask) so numbers match training-time reporting;
    both Isaac Sim and Gymnasium here use NextStep auto-reset."""
    device = agent.device
    state_norm = getattr(agent, "state_normalizer", None)
    goal_norm = getattr(agent, "goal_normalizer", None)
    if state_norm is not None:
        state_norm.eval()
    if goal_norm is not None:
        goal_norm.eval()

    obs = env.reset(seed=seed)
    states, goals = obs.states, obs.goals
    num_envs = env.num_envs
    prev_done = T.zeros(num_envs, dtype=T.bool, device=device)
    accum = T.zeros(num_envs, dtype=T.float32, device=device)
    scores = []

    while len(scores) < num_episodes:
        norm_states = state_norm.normalize(states) if state_norm is not None else states
        norm_goals = goal_norm.normalize(goals) if (goal_norm is not None and goals is not None) else goals
        actions = select_action(agent, norm_states, norm_goals, deterministic)

        obs = env.step(actions)
        rewards = obs.rewards.flatten().to(device)
        dones = T.logical_or(obs.terminations, obs.truncations).flatten().to(device)

        valid = ~prev_done
        accum[valid] += rewards[valid]

        for i in dones.nonzero(as_tuple=False).flatten().tolist():
            scores.append(float(accum[i].item()))
            logger.info("Episode %d (env %d): reward=%.3f | avg=%.3f",
                        len(scores), i, scores[-1], sum(scores) / len(scores))
            accum[i] = 0.0
            if len(scores) >= num_episodes:
                break

        prev_done = dones
        states, goals = obs.states, obs.goals

    return scores


def main():
    parser = argparse.ArgumentParser(description="Watch / evaluate a trained agent.")
    parser.add_argument("--agent_dir", required=True,
                        help="Saved agent directory (contains config.json + policy/).")
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes to run before exiting.")
    parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs to run/display (default 1).")
    parser.add_argument("--render_mode", type=str, default="human",
                        help="Isaac Sim: any value != 'headless' opens the GUI. Gymnasium: 'human' opens a window.")
    parser.add_argument("--seed", type=int, default=None, help="Override the env seed.")
    # parser.add_argument("--stochastic", action="store_true",
                        # help="Sample actions instead of using the deterministic mean action.")
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    if not (agent_dir / "config.json").exists():
        raise FileNotFoundError(f"No config.json in {agent_dir}")

    env = build_eval_env(agent_dir, args.num_envs, args.render_mode, args.seed)
    try:
        trainer = load_trainer_from_dir(agent_dir, env=env)
        scores = run(agent, env, args.num_episodes, deterministic=not args.stochastic, seed=args.seed)
        if scores:
            logger.info("Done. %d episodes | mean %.3f | min %.3f | max %.3f",
                        len(scores), sum(scores) / len(scores), min(scores), max(scores))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
