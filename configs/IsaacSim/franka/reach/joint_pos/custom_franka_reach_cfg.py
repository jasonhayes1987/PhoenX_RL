# custom_franka_reach_cfg.py
#
# Subclass of the official Franka Reach environment that lets you easily
# control the joint position action scale.
#
# Recommended for SAC + custom SquashedNormal:
#   - Keep your distribution output roughly in [-1, 1] or [-2, 2]
#   - Increase action_scale to 2.0–3.0 (or higher) for better reachability
#   - This gives effective joint deltas of ~ ±1.0 to ±1.5 rad per step

import sys
import os

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
    FrankaReachEnvCfg_PLAY,
)


@configclass
class FrankaReachEnvCfg_Custom(FrankaReachEnvCfg):
    """Custom Franka Reach config with adjustable joint position action scale.

    Use this when training with SAC (or any squashed policy) so you can
    increase the effective step size in joint space without making your
    SquashedNormal distribution output extremely large values.
    """

    action_scale: float = 2.0
    """Scale applied to policy actions before adding to default joint positions.

    Original default = 0.5.
    Recommended range for SAC reach tasks: 1.5 – 3.0 (start with 2.0).
    Higher values = larger joint deltas per step → easier to reach targets.
    """

    def __post_init__(self):
        # Always call parent first
        super().__post_init__()

        # Override ONLY the arm action with scale
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=self.action_scale,
            use_default_offset=True,
            # Optional safety clip on the raw policy output (before env scale)
            # clip={".*": (-3.0, 3.0)},
        )


@configclass
class FrankaReachEnvCfg_Custom_PLAY(FrankaReachEnvCfg_Custom):
    """Play / evaluation version with fewer environments and no corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# Optional: even more aggressive version (maps [-1,1] directly to joint limits)
@configclass
class FrankaReachEnvCfg_Custom_Limits(FrankaReachEnvCfg):
    """Alternative version using JointPositionToLimitsActionCfg.
    Policy output in ~[-1, 1] directly spans the full physical joint range.
    Can be more aggressive / less stable than the scaled delta version.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            rescale_to_limits=True,
        )