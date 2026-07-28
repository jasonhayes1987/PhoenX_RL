"""Watch / evaluate a trained PhoenX_RL policy in an Isaac Sim GUI.

Thin wrapper over the unified loader: it rebuilds the Isaac Sim env from the
saved run ``config.json`` (GUI on, fewer envs, training-only wrappers dropped)
and hands that live env to :meth:`Trainer.load`, so the exact saved agent
(policy weights + frozen normalizers) is restored and rolled out.

Run it from inside the ``rl_env`` conda env:

    conda activate rl_env
    python src/scripts/play_isaacsim.py --agent_dir "src/Trained_Models/IsaacSim/Franka/Reach/JointPos/SAC_1" --num_envs 16
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phoenx.trainer import Trainer
from phoenx.test import build_eval_env


def main():
    parser = argparse.ArgumentParser(description="Watch a trained policy act in an Isaac Sim GUI.")
    parser.add_argument("--agent_dir", required=True, help="Saved agent dir, e.g. .../JointPos/SAC_1")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs to display (default 16).")
    parser.add_argument("--episodes", type=int, default=20, help="Stop after this many completed episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override the env seed.")
    parser.add_argument("--render_mode", type=str, default="human",
                        help="Any value except 'headless' opens the GUI window.")
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    if not (agent_dir / "config.json").exists():
        raise FileNotFoundError(f"No config.json in {agent_dir}")

    # Build the (single) Isaac Sim env first, then let Trainer.load reuse it.
    env = build_eval_env(agent_dir, args.num_envs, args.render_mode, args.seed)
    try:
        trainer = Trainer.load(agent_dir, env=env, load_weights=True)
        trainer.test(unit="episode", units=args.episodes)
    finally:
        try:
            env.close()
        except Exception as e:
            raise e


if __name__ == "__main__":
    main()
