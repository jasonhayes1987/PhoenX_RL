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

    python src/phoenx/cli/test.py --agent_dir "src/Trained_Models/IsaacSim/Franka/Reach/JointPos/SAC_3" --num_envs 1 --num_episodes 10
    python src/phoenx/cli/test.py --agent_dir "path/to/LunarLanderContinuous-v3/SAC_1" --render_mode human
"""

import json
import argparse
from pathlib import Path

from phoenx.env_wrapper import EnvWrapper
from phoenx.trainer import Trainer
from phoenx.logging_config import configure_logging


# Training-only wrappers that must not run during evaluation: they require a
# per-step ``set_action`` call and only build n-step trajectories for the buffer.
# They do not change the observation/action spaces, so dropping them is safe.
_TRAIN_ONLY_WRAPPERS = {"VectorNStepReward"}


def build_eval_env(agent_dir: Path, env: str|None, num_envs: int|None, render_mode: str|None, seed: int|None) -> EnvWrapper:
    """Rebuild the env from the saved run config, overridden for viewing."""
    with open(agent_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    spec = dict(config["env"])
    env_cfg = dict(spec["config"])
    if env is not None:
        env_cfg["cfg"] = env
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch / evaluate a trained agent.")
    parser.add_argument(
        "--agent_dir",
        required=True,
        help="Saved agent directory (contains config.json + policy/)."
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override the env cfg, e.g. "
             "'Configs.IsaacSim.franka.cube_lift.custom_franka_cube_lift_cfg:FrankaCubeLiftEnvCfg_Custom_PLAY'. "
             "For Gymnasium agents this is the env id, e.g. 'LunarLanderContinuous-v3'."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Episodes to run before exiting."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Parallel envs to run/display (default 1)."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="Isaac Sim: any value != 'headless' opens the GUI. Gymnasium: 'human' opens a window."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the env seed."
    )
    parser.add_argument(
        '--log_level',
        default='INFO',
        type=str,
        required=False,
        help='Logging level'
    )
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    if not (agent_dir / "config.json").exists():
        raise FileNotFoundError(f"No config.json in {agent_dir}")

    # Create logger
    logger = configure_logging(args.log_level, log_dir=agent_dir)

    env = build_eval_env(agent_dir, args.env, args.num_envs, args.render_mode, args.seed)
    
    trainer = Trainer.load(agent_dir, env=env, load_weights=True, load_buffer=False, log_level=args.log_level)
    trainer.test(unit="episode", units=args.num_episodes)
    
if __name__ == "__main__":
    main()
