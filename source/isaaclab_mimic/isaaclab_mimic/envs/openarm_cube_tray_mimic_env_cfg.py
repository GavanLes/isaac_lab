"""
Environment configuration for OpenArmX Cube-to-Tray pick-and-place with Mimic data generation.

Follows the pattern of FrankaCubeStackEnvCfg + MimicEnvCfg but adapted for:
- OpenArmX bimanual robot (left arm performs the task)
- Single green cube (pick source)
- Single tray (place target)
"""

from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.envs.mdp import image, last_action, time_out


# ---------------------------------------------------------------------------
# Custom spawner: flat UsdGeomCube with all APIs on the same prim
# (EXACT replica of create_dynamic_cube() in scene_utils.py using raw pxr calls)
# ---------------------------------------------------------------------------

@clone
def _spawn_direct_cube(
    prim_path: str,
    cfg: "DirectCubeCfg",
    translation: tuple | None = None,
    orientation: tuple | None = None,
    **kwargs,
):
    """Spawn a dynamic cube matching create_dynamic_cube() byte-for-byte.

    Uses raw pxr calls (not Isaac Lab helpers) to avoid any subtle property
    differences that could affect physics behavior.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    z_scale = cfg.height / cfg.size

    # --- Exactly create_dynamic_cube() lines 203-212 ---
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(float(cfg.size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.8, 0.0)])

    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, float(z_scale)))

    # --- Combined material (visual + physics) like create_preview_material() lines 147-183 ---
    mat_path = f"{prim_path}/material"
    material = UsdShade.Material.Define(stage, mat_path)

    shader = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.8, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr(20.0)
    physics_mat.CreateDynamicFrictionAttr(20.0)
    physics_mat.CreateRestitutionAttr(0.0)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_mat.CreateFrictionCombineModeAttr("max")

    # --- All-purpose material binding like bind_material() line 187 ---
    UsdShade.MaterialBindingAPI(prim).Bind(material)

    # --- Collision, rigid body, mass exactly like create_dynamic_cube() lines 224-248 ---
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)

    rigid_api = UsdPhysics.RigidBodyAPI(prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(False)
    rigid_api.CreateStartsAsleepAttr(False)

    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
    else:
        mass_api = UsdPhysics.MassAPI(prim)
    mass_api.CreateMassAttr(0.05)
    # Match recording: do not author inertia/COM. A stale explicit inertia
    # can make the cube yaw/tilt unnaturally while grasped.
    for attr in (
        mass_api.GetDiagonalInertiaAttr(),
        mass_api.GetPrincipalAxesAttr(),
        mass_api.GetCenterOfMassAttr(),
        mass_api.GetDensityAttr(),
    ):
        if attr and attr.HasAuthoredValueOpinion():
            attr.Clear()

    return prim


@configclass
class DirectCubeCfg(SpawnerCfg):
    """Configuration for spawning a dynamic cube matching create_dynamic_cube().

    All physics/material properties are hard-coded inside _spawn_direct_cube
    to match the recording exactly — only geometric size/height are configurable.
    """

    size: float = 0.04
    height: float = 0.08

    def __post_init__(self):
        self.func = _spawn_direct_cube


# Scene position constants (must match scene_config.py and annotate_demos.py)
# Tray fixed world position from recording scene
_TRAY_POS = (-1.71578, -0.0800, 0.1797)
# Robot base position inferred from recording:
#   cube_init = robot_pos + choose_front_object_position offset (0.35, 0.2, 0.35)
#   cube_init = [-1.733, 0.200, 0.315]  =>  robot_pos ≈ [-2.083, 0.0, -0.035]
_ROBOT_POS = (-2.083, 0.0, 0.0)
_CUBE_START_Z = 0.315
_GRIPPER_HOLD_POS = 0.0
_ARM_VELOCITY_LIMIT_SIM = 100000.0
_ARM_EFFORT_LIMIT_SIM = 1000000000.0
_ARM_STIFFNESS = 100000.0
_ARM_DAMPING = 2000.0
# Path to recording scene USD
_ENV_USD_PATH = "/home/huatec/isaac_lab/my_env/environment.usd"
_OPENARMX_USD_PATH = "/home/huatec/isaac_lab/my_env/openarmx.usd"

# Custom robot ArticulationCfg using openarmx.usd (18 DOF, built-in cameras)
# Based on OPENARM_BI_HIGH_PD_CFG but with openarmx joint naming
_OPENARMX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_OPENARMX_USD_PATH,
        # No rigid_props / articulation_props overrides — match recording exactly
        # Recording loads openarmx.usd directly with NO physics property overrides
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarmx_left_joint.*": 0.0,
            "openarmx_right_joint.*": 0.0,
            "openarmx_left_finger_joint.*": 0.0,
            "openarmx_right_finger_joint.*": 0.0,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarmx_left_joint[1-7]",
                "openarmx_right_joint[1-7]",
            ],
            velocity_limit_sim=_ARM_VELOCITY_LIMIT_SIM,
            effort_limit_sim=_ARM_EFFORT_LIMIT_SIM,
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            armature=0.0,
            friction=0.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarmx_left_finger_joint.*",
                "openarmx_right_finger_joint.*",
            ],
            velocity_limit_sim=1.0,
            effort_limit_sim=333.33,
            stiffness=5e4,
            damping=5e2,
            armature=0.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=5000.0,
)

# ---------------------------------------------------------------------------
# Standalone MDP functions
# ---------------------------------------------------------------------------

# Default OpenArm BI joint poses
_OPENARM_RIGHT_DEFAULT = [-0.31782240, -0.02627741, -0.02781186, 0.97955275, 0.02013962, -0.00748043, 0.80385858, 0.0, 0.0]
_OPENARM_LEFT_DEFAULT = [0.36884275, 0.00019181, -0.00019181, 1.04745209, -0.00287709, 0.03394964, -0.79350102, 0.0, 0.0]
_OPENARM_DEFAULT_ALL = _OPENARM_RIGHT_DEFAULT + _OPENARM_LEFT_DEFAULT
# Joint ordering for OpenArm BI: right_joint[1-7], right_finger[1-2], left_joint[1-7], left_finger[1-2]


def _joint_pos_rel_left(env):
    """Relative joint positions for left arm (7 joints)."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("openarmx_left_joint[1-7]")
    return robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]


def _joint_vel_rel_left(env):
    """Relative joint velocities for left arm (7 joints)."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("openarmx_left_joint[1-7]")
    return robot.data.joint_vel[:, joint_ids]


def _cube_pos_w(env):
    """Cube position in world frame (relative to env origin)."""
    pos = env.scene["cube"].data.root_pos_w
    return pos - env.scene.env_origins


def _eef_pos_w(env):
    """Left hand position in world frame (relative to env origin)."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("openarmx_left_hand")
    pos = robot.data.body_link_state_w[:, body_ids[0], :3]
    return pos - env.scene.env_origins


def _eef_quat_w(env):
    """Left hand quaternion in world frame."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("openarmx_left_hand")
    return robot.data.body_link_state_w[:, body_ids[0], 3:7]


def _object_grasped(env, robot_cfg, object_cfg, diff_threshold=0.06):
    """Check if the cube is grasped by the left gripper (uses body state, not FrameTransformer)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    body_ids, _ = robot.find_bodies("openarmx_left_hand")
    eef_pos = robot.data.body_link_state_w[:, body_ids[0], :3]
    obj_pos = obj.data.root_pos_w
    pose_diff = torch.linalg.vector_norm(obj_pos - eef_pos, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]
                  - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device))
        > env.cfg.gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]
                  - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device))
        > env.cfg.gripper_threshold,
    )
    return grasped


def _cube_in_tray(env):
    """Whether cube centre is within tray XY bounds and above its base."""
    cube = env.scene["cube"]
    cp = cube.data.root_pos_w
    tp = torch.tensor(_TRAY_POS, device=cp.device, dtype=cp.dtype)
    inside_xy = torch.logical_and(
        torch.abs(cp[:, 0] - tp[0]) < 0.07,
        torch.abs(cp[:, 1] - tp[1]) < 0.105,
    )
    above_base = cp[:, 2] - tp[2] > 0.002
    below_top = cp[:, 2] - tp[2] < 0.045
    return torch.logical_and(torch.logical_and(inside_xy, above_base), below_top).float()


def _cube_below_ground(env):
    return env.scene["cube"].data.root_pos_w[:, 2] < -0.05


def _gripper_pos(env, robot_cfg):
    """Left gripper joint positions (mean of both fingers)."""
    robot = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    return robot.data.joint_pos[:, joint_ids].mean(dim=-1, keepdim=True)


def _task_success(env):
    return _cube_in_tray(env).bool()


# Event functions


def _reset_default_joint_pose(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Set default joint pose for all OpenArm BI joints on reset."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    # Reset velocities and targets
    asset.data.joint_vel[env_ids] = 0.0
    asset.data.joint_acc[env_ids] = 0.0
    # Set joint positions for each known joint pattern (robot may have extra DOFs)
    default_map = {
        "openarmx_left_joint[1-7]": _OPENARM_LEFT_DEFAULT[:7],
        "openarmx_right_joint[1-7]": _OPENARM_RIGHT_DEFAULT[:7],
        "openarmx_left_finger_joint.*": _OPENARM_LEFT_DEFAULT[7:9],
        "openarmx_right_finger_joint.*": _OPENARM_RIGHT_DEFAULT[7:9],
    }
    for pattern, values in default_map.items():
        jids, _ = asset.find_joints(pattern)
        if len(jids) == 0:
            continue
        for jid, val in zip(jids, values):
            asset.data.joint_pos[env_ids, jid] = val
            asset.data.default_joint_pos[env_ids, jid] = val
    asset.write_joint_state_to_sim(
        position=asset.data.joint_pos[env_ids],
        velocity=torch.zeros(n, asset.data.joint_pos.shape[1], device=env.device),
        env_ids=env_ids,
    )
    asset.set_joint_position_target(asset.data.joint_pos[env_ids], env_ids=env_ids)
    asset.set_joint_velocity_target(
        torch.zeros(n, asset.data.joint_pos.shape[1], device=env.device),
        env_ids=env_ids,
    )
    if hasattr(env, "_right_hold_eef_pose"):
        env._right_hold_eef_pose = None


def _randomize_cube_pose(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")):
    """Randomize cube pose robot-relative, matching recording's choose_front_object_position."""
    import random
    asset = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[env_ids, :3]
    for idx, i in enumerate(env_ids.tolist()):
        rx = robot_pos[idx, 0].item()
        ry = robot_pos[idx, 1].item()
        # Match recording's height exactly. Randomizing Z can spawn the cube
        # partially into the table/tray contact shell and corrupt its pose.
        x = rx + 0.35 + random.uniform(-0.05, 0.05)
        y = ry + 0.20 + random.uniform(-0.05, 0.05)
        z = _CUBE_START_Z
        pos = torch.tensor([[x, y, z]], device=env.device)
        asset.write_root_pose_to_sim(
            torch.cat([pos, torch.tensor([[1, 0, 0, 0]], device=env.device)], dim=-1),
            env_ids=torch.tensor([i], device=env.device),
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([i], device=env.device),
        )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class CubeTraySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    cube: RigidObjectCfg = MISSING

    # Recording environment (table, room, lights, tray)
    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=UsdFileCfg(usd_path=_ENV_USD_PATH),
    )


# ---------------------------------------------------------------------------
# MDP Configs
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    right_arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    right_gripper_action: BinaryJointPositionActionCfg = MISSING
    left_arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    left_gripper_action: BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(func=_joint_pos_rel_left)
        left_joint_vel = ObsTerm(func=_joint_vel_rel_left)
        eef_pos = ObsTerm(func=_eef_pos_w)
        eef_quat = ObsTerm(func=_eef_quat_w)
        gripper_pos = ObsTerm(func=_gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        cube_pos = ObsTerm(func=_cube_pos_w)
        actions = ObsTerm(func=last_action, params={"action_name": "left_arm_action"})
        left_hand_cam = ObsTerm(
            func=image, params={"sensor_cfg": SceneEntityCfg("left_hand_cam"), "data_type": "rgb", "normalize": False}
        )
        right_hand_cam = ObsTerm(
            func=image, params={"sensor_cfg": SceneEntityCfg("right_hand_cam"), "data_type": "rgb", "normalize": False}
        )
        body_cam = ObsTerm(
            func=image, params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskTermsCfg(ObsGroup):
        grasp = ObsTerm(func=_object_grasped, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube"),
        })
        place = ObsTerm(func=_cube_in_tray)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskTermsCfg = SubtaskTermsCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    cube_dropped = DoneTerm(func=_cube_below_ground)
    success = DoneTerm(func=_task_success)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(func=_reset_default_joint_pose, mode="reset")
    randomize_cube = EventTerm(func=_randomize_cube_pose, mode="reset")


# ---------------------------------------------------------------------------
# Base RL Env Config
# ---------------------------------------------------------------------------


@configclass
class OpenArmCubeTrayEnvCfg(ManagerBasedRLEnvCfg):
    scene: CubeTraySceneCfg = CubeTraySceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    gripper_joint_names: list = MISSING
    gripper_open_val: float = 0.04
    gripper_close_val: float = _GRIPPER_HOLD_POS
    gripper_threshold: float = 0.005

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 30.0
        # Match record_demo.py exactly. The cube/gripper contact is sensitive
        # to CPU vs GPU PhysX differences during grasped lateral motion.
        self.sim.device = "cpu"
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        # Camera matches recording (scene_config.CAMERA_EYE)
        self.viewer.eye = (-2.6, -1.07, 0.85)
        self.viewer.lookat = (-1.21, 0.18, 0.18)
        # --- Scene-level PhysX settings matched to record_demo.py ---
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025  # PhysX default (recording doesn't override)
        self.sim.physx.enable_ccd = True
        self.sim.physx.solve_articulation_contact_last = True
        # Position iterations: recording uses min=8 / max=64; we match exactly
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.max_position_iteration_count = 64
        # Velocity iterations: recording uses min=1 / max=32
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.max_velocity_iteration_count = 32

        # Robot — positioned to match recording's openarmx.usd prim world position
        self.scene.robot = _OPENARMX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = _ROBOT_POS

        # Cube — exact replica of create_dynamic_cube() in scene_utils.py
        # All physics, material, and API properties are hardcoded inside _spawn_direct_cube
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(_ROBOT_POS[0] + 0.35, _ROBOT_POS[1] + 0.2, _CUBE_START_Z), rot=[1, 0, 0, 0],
            ),
            spawn=DirectCubeCfg(size=0.04, height=0.08),
        )

        # IK actions — both arms defined, left is the active one for the task
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["openarmx_right_joint[1-7]"],
            body_name="openarmx_right_hand", controller=ik_cfg, scale=1.0,
        )
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["openarmx_left_joint[1-7]"],
            body_name="openarmx_left_hand", controller=ik_cfg, scale=1.0,
        )

        self.actions.right_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarmx_right_finger_joint.*"],
            open_command_expr={"openarmx_right_finger_joint.*": 0.04},
            close_command_expr={"openarmx_right_finger_joint.*": _GRIPPER_HOLD_POS},
        )
        self.actions.left_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarmx_left_finger_joint.*"],
            open_command_expr={"openarmx_left_finger_joint.*": 0.04},
            close_command_expr={"openarmx_left_finger_joint.*": _GRIPPER_HOLD_POS},
        )

        self.gripper_joint_names = ["openarmx_left_finger_joint.*"]


# ---------------------------------------------------------------------------
# Mimic Env Config
# ---------------------------------------------------------------------------


@configclass
class OpenArmCubeTrayMimicEnvCfg(OpenArmCubeTrayEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Cameras from openarmx.usd (reference existing prims, no spawn needed)
        self.scene.left_hand_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarmx_left_hand/left_Camera",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
            ),
        )

        self.scene.right_hand_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarmx_right_hand/right_Camera",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
            ),
        )

        self.scene.body_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarmx_body_link0/hand_Camera",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
            ),
        )

        # Camera rendering settings
        self.num_rerenders_on_reset = 3
        self.image_obs_list = ["left_hand_cam", "right_hand_cam", "body_cam"]

        self.datagen_config = DataGenConfig(
            generation_guarantee=True,
            generation_keep_failed=False,
            max_num_failures=50,
            seed=1,
            generation_num_trials=100,
            generation_select_src_per_subtask=True,
            generation_select_src_per_arm=False,
            generation_transform_first_robot_pose=True,
            generation_interpolate_from_last_target_pose=True,
        )

        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="cube", subtask_term_signal="grasp",
                selection_strategy="nearest_neighbor_object",
                subtask_term_offset_range=(0, 5),
                action_noise=0.0, num_interpolation_steps=8, num_fixed_steps=8,
                description="Approach and grasp the cube",
            ),
            SubTaskConfig(
                object_ref="tray", subtask_term_signal="place",
                selection_strategy="nearest_neighbor_object",
                subtask_term_offset_range=(0, 5),
                action_noise=0.0, num_interpolation_steps=16, num_fixed_steps=8,
                description="Move cube to tray and release",
            ),
            SubTaskConfig(
                object_ref=None, subtask_term_signal=None,
                selection_strategy="random",
                subtask_term_offset_range=(0, 0),
                action_noise=0.0, num_interpolation_steps=3, num_fixed_steps=0,
                description="Retreat (final subtask)",
            ),
        ]
