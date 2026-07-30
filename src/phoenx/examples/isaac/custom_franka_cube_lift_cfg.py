# Subclass of the official Franka Cube Lift environment that lets you easily
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

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    ObservationsCfg as LiftObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)


@configclass
class FrankaCubeLiftEnvCfg_Custom(FrankaCubeLiftEnvCfg):
    """Custom Franka Cube Lift config with adjustable joint position action scale."""

    action_scale: float = 0.5
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

        # Remove action rate and joint velocity curriculum
        # self.curriculum.action_rate = None
        # self.curriculum.joint_vel = None


@configclass
class FrankaCubeLiftEnvCfg_Custom_PLAY(FrankaCubeLiftEnvCfg_Custom):
    """Play / evaluation version with fewer environments and no corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# Optional: even more aggressive version (maps [-1,1] directly to joint limits)
@configclass
class FrankaCubeLiftEnvCfg_Custom_Limits(FrankaCubeLiftEnvCfg):
    """Alternative version using JointPositionToLimitsActionCfg.
    Policy output in ~[-1, 1] directly spans the full physical joint range.
    Can be more aggressive / less stable than the scaled delta version.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            rescale_to_limits=True,
        )


@configclass
class FrankaCubeLiftEnvCfg_Custom_Limits_PLAY(FrankaCubeLiftEnvCfg_Custom_Limits):
    """Play / evaluation version with fewer environments and no corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

# Sparse Reward (gaol conditioned) config
def target_object_position_b(env, command_name: str = "object_pose"):
    """Position slice of the commanded object pose (drops the quaternion)."""
    return env.command_manager.get_command(command_name)[:, :3]


@configclass
class GoalObservationsCfg(LiftObservationsCfg):
    """Lift observations plus separate achieved/desired goal groups for HER."""
    @configclass
    class AchievedGoalCfg(ObsGroup):
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class DesiredGoalCfg(ObsGroup):
        target_position = ObsTerm(func=target_object_position_b)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()
    desired_goal: DesiredGoalCfg = DesiredGoalCfg()


@configclass
class FrankaCubeLiftEnvCfg_Custom_Goal(FrankaCubeLiftEnvCfg_Custom_Limits):
    observations: GoalObservationsCfg = GoalObservationsCfg()

@configclass
class FrankaCubeLiftEnvCfg_Custom_Goal_PLAY(FrankaCubeLiftEnvCfg_Custom_Goal):

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# =============================================================================
# Multi-modal (camera + proprioception) variants
# =============================================================================
# Adds a per-env table-view TiledCamera and exposes it as its OWN observation
# group ('rgb'), alongside the standard 'policy' state group. With the PhoenX
# wrapper configured as `obs_key: null`, the agent receives a dict observation
#   { 'policy': (N, 36) float32, 'rgb': (N, H, W, 3) uint8 }
# which maps directly onto the roots -> trunk -> branches architecture
# (see phoenx/examples/configs/IsaacSim/franka/cube_lift/dense/ppo_camera.yml).
#
# NOTE: requires `enable_cameras: true` in the env config so the Kit app
# launches with tiled rendering.

CAMERA_WIDTH = 84
CAMERA_HEIGHT = 84


@configclass
class CameraObservationsCfg(LiftObservationsCfg):
    """Lift observations plus a separate 'rgb' image group."""

    @configclass
    class RGBCfg(ObsGroup):
        """Raw RGB frames from the table camera (uint8, channels-last).

        normalize=False keeps uint8 so the rollout buffer stores images
        compactly; the model casts/scales at its input boundary.
        """

        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb",
                    "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    rgb: RGBCfg = RGBCfg()


@configclass
class FrankaCubeLiftCameraEnvCfg(FrankaCubeLiftEnvCfg_Custom_Limits):
    """Cube lift with full state + a table-view RGB camera (pipeline check).

    The state group still contains object_position, so this variant verifies
    the multi-modal plumbing without REQUIRING vision to solve the task.
    """

    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Per-env tiled camera looking at the table workspace (pose adapted
        # from IsaacLab's Franka stack visuomotor table_cam).
        self.scene.table_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=32.0, clipping_range=(0.1, 2.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.3, 0.0, 0.4), rot=(0.35355, -0.43534, -0.43534, 0.35355),
                convention="ros",
            ),
        )

        # Remove action rate and joint velocity curriculum
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # Fresh camera frames for the spliced terminal/reset observations the
        # NextStep autoreset conversion captures.
        self.rerender_on_reset = True
        # Disable DLSS antialiasing for tiled-render throughput.
        self.sim.render.antialiasing_mode = "OFF"


@configclass
class FrankaCubeLiftCameraBlindEnvCfg(FrankaCubeLiftCameraEnvCfg):
    """Camera cube lift with object_position REMOVED from the state group.

    The cube's location is only observable through the camera — the true test
    that the vision root carries task-relevant information.
    """

    def __post_init__(self):
        super().__post_init__()
        # The policy state keeps joints + target command + last action, but
        # loses the ground-truth object position (36 -> 33 features).
        self.observations.policy.object_position = None


@configclass
class FrankaCubeLiftCameraEnvCfg_PLAY(FrankaCubeLiftCameraEnvCfg):
    """Play / evaluation version with fewer environments and no corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftCameraBlindEnvCfg_PLAY(FrankaCubeLiftCameraBlindEnvCfg):
    """Play / evaluation version with fewer environments and no corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
