"""
Mimic environment for OpenArmX Cube-to-Tray pick-and-place task.

Implements ManagerBasedRLMimicEnv APIs for MimicGen data generation.

Action space (14-dim) used during generation (matches ActionsCfg declaration order):
  [0:6]   right arm delta pose  (Differential IK, holds reset pose)
  [6]     right gripper          (Binary: fixed close -> 0)
  [7:13]  left arm delta pose   (Differential IK, scale=0.5, active)
  [13]    left gripper           (Binary: >=0 open, <0 close)

Recorded actions (18-dim joint positions) are also handled by
actions_to_gripper_actions for datagen info pool loading:
  [0:7]   right_joint[1-7], [7:9] right_finger[1-2],
  [9:16]  left_joint[1-7],  [16:18] left_finger[1-2]
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class OpenArmCubeTrayMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for the OpenArmX Cube-to-Tray task.

    Left arm picks a green cube and places it into a tray.
    Right arm stays at its default idle configuration.
    """

    def __init__(self, cfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode=render_mode)
        self._right_hold_eef_pose = None
        # Friction is now baked into CuboidCfg.physics_material during spawn
        # _apply_cube_friction removed — post-spawn material changes can disrupt PhysX

    def _apply_cube_friction(self):
        """Verify cube material friction after scene creation (safe post-processing).

        Does NOT modify collision geometry/solver settings — those are baked in
        at spawn time by _spawn_direct_cube to avoid tensor view invalidation.
        """
        from pxr import PhysxSchema, UsdPhysics, UsdShade
        from isaaclab.sim.utils.stage import get_current_stage

        def _walk(prim):
            yield prim
            for c in prim.GetChildren():
                yield from _walk(c)

        stage = get_current_stage()
        for env_path in self.scene.env_prim_paths:
            cube_path = f"{env_path}/Cube"
            cube_prim = stage.GetPrimAtPath(cube_path)
            if not cube_prim.IsValid():
                continue

            # Update material friction on existing materials (doesn't change geometry)
            for child in _walk(cube_prim):
                mb = UsdShade.MaterialBindingAPI(child)
                mat, _ = mb.ComputeBoundMaterial()
                if not mat:
                    continue
                mat_prim = mat.GetPrim()
                if not mat_prim.IsValid():
                    continue
                api = UsdPhysics.MaterialAPI.Apply(mat_prim)
                api.CreateStaticFrictionAttr().Set(20.0)
                api.CreateDynamicFrictionAttr().Set(20.0)
                api.CreateRestitutionAttr().Set(0.0)
                px_api = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
                px_api.CreateFrictionCombineModeAttr("max")

    # Action dimension layout for 14-dim delta-pose format:
    # MUST match ActionsCfg field declaration order:
    #   right_arm_action[0:6], right_gripper_action[6:7],
    #   left_arm_action[7:13], left_gripper_action[13:14]
    _ra = slice(0, 6)       # right arm IK delta (idle)
    _rg = slice(6, 7)       # right gripper binary
    _la = slice(7, 13)      # left arm IK delta (active)
    _lg = slice(13, 14)     # left gripper binary

    # Gripper threshold for converting 18-dim finger joint positions to binary
    _GRIP_THRESHOLD = 0.03

    def get_robot_eef_pose(
        self,
        eef_name: str = "left",
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return (N, 4, 4) EEF pose matrix in world frame from articulation body state."""
        robot = self.scene["robot"]
        body_ids, _ = robot.find_bodies(f"openarmx_{eef_name}_hand")
        if env_ids is None:
            env_ids = slice(None)
        state = robot.data.body_link_state_w[env_ids, body_ids[0], :7]
        pos = state[..., :3]
        quat = state[..., 3:7]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses + gripper actions -> flat (1, 14) action tensor.

        Layout (must match ActionsCfg declaration order):
          [0:6] right arm, [6] right gripper, [7:13] left arm, [13] left gripper.
        """
        device = self.device

        # Right arm: hold the reset pose. A zero relative IK command follows
        # the current pose, so gravity drift would become the new target.
        if "right" in target_eef_pose_dict:
            target_r = target_eef_pose_dict["right"]
        else:
            if self._right_hold_eef_pose is None:
                self._right_hold_eef_pose = self.get_robot_eef_pose(eef_name="right").detach().clone()
            target_r = self._right_hold_eef_pose[env_id]
        curr_r = self.get_robot_eef_pose(eef_name="right", env_ids=[env_id])[0]
        delta_pos_r = target_r[:3, 3] - curr_r[:3, 3]
        delta_rot_mat_r = target_r[:3, :3] @ curr_r[:3, :3].T
        delta_rot_quat_r = PoseUtils.quat_from_matrix(delta_rot_mat_r)
        delta_rot_aa_r = PoseUtils.axis_angle_from_quat(delta_rot_quat_r)
        right_pose = torch.cat([delta_pos_r, delta_rot_aa_r])
        if action_noise_dict is not None and "right" in action_noise_dict:
            right_pose += action_noise_dict["right"] * torch.randn_like(right_pose)
        # Right gripper stays closed at 0 throughout the generated trajectory.
        right_grip = torch.tensor([-1.0], device=device)

        # Left arm (active) delta pose
        target_l = target_eef_pose_dict["left"]
        curr_l = self.get_robot_eef_pose(eef_name="left", env_ids=[env_id])[0]
        delta_pos_l = target_l[:3, 3] - curr_l[:3, 3]
        delta_rot_mat_l = target_l[:3, :3] @ curr_l[:3, :3].T
        delta_rot_quat_l = PoseUtils.quat_from_matrix(delta_rot_mat_l)
        delta_rot_aa_l = PoseUtils.axis_angle_from_quat(delta_rot_quat_l)
        left_pose = torch.cat([delta_pos_l, delta_rot_aa_l])  # (6,)

        if action_noise_dict is not None and "left" in action_noise_dict:
            left_pose += action_noise_dict["left"] * torch.randn_like(left_pose)

        left_grip = gripper_action_dict.get("left", torch.tensor(1.0, device=device))
        if left_grip.ndim == 0:
            left_grip = left_grip.reshape(1)

        action = torch.cat([right_pose, right_grip, left_pose, left_grip])
        return action.unsqueeze(0)  # (1, 14)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract target EEF poses from a delta-pose action tensor (inverse of _to_action).

        Layout (must match ActionsCfg): right_arm[0:6], right_grip[6], left_arm[7:13], left_grip[13].
        """
        # Right arm (idle, indices 0:6)
        delta_pos_r = action[..., 0:3]
        delta_rot_aa_r = action[..., 3:6]
        delta_rot_angle_r = torch.linalg.norm(delta_rot_aa_r, dim=-1)
        delta_rot_axis_r = delta_rot_aa_r / delta_rot_angle_r.clamp(min=1e-12).unsqueeze(-1)
        zero_angle_r = torch.isclose(delta_rot_angle_r, torch.zeros_like(delta_rot_angle_r))
        delta_rot_axis_r[zero_angle_r] = torch.zeros(3, device=delta_rot_aa_r.device, dtype=delta_rot_aa_r.dtype)
        delta_rot_quat_r = PoseUtils.quat_from_angle_axis(delta_rot_angle_r, delta_rot_axis_r)
        delta_rot_mat_r = PoseUtils.matrix_from_quat(delta_rot_quat_r)

        curr_r = self.get_robot_eef_pose(eef_name="right")
        target_r_pos = curr_r[..., :3, 3] + delta_pos_r
        target_r_rot = delta_rot_mat_r @ curr_r[..., :3, :3]
        target_r = PoseUtils.make_pose(target_r_pos, target_r_rot)

        # Left arm (active, indices 7:13)
        delta_pos_l = action[..., 7:10]
        delta_rot_aa_l = action[..., 10:13]
        delta_rot_angle_l = torch.linalg.norm(delta_rot_aa_l, dim=-1)
        delta_rot_axis_l = delta_rot_aa_l / delta_rot_angle_l.clamp(min=1e-12).unsqueeze(-1)
        zero_angle_l = torch.isclose(delta_rot_angle_l, torch.zeros_like(delta_rot_angle_l))
        delta_rot_axis_l[zero_angle_l] = torch.zeros(3, device=delta_rot_aa_l.device, dtype=delta_rot_aa_l.dtype)
        delta_rot_quat_l = PoseUtils.quat_from_angle_axis(delta_rot_angle_l, delta_rot_axis_l)
        delta_rot_mat_l = PoseUtils.matrix_from_quat(delta_rot_quat_l)

        curr_l = self.get_robot_eef_pose(eef_name="left")
        target_l_pos = curr_l[..., :3, 3] + delta_pos_l
        target_l_rot = delta_rot_mat_l @ curr_l[..., :3, :3]
        target_l = PoseUtils.make_pose(target_l_pos, target_l_rot)

        return {"left": target_l, "right": target_r}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract gripper values from action sequences.

        Handles both 18-dim recorded joint positions and 14-dim delta-pose actions.

        Args:
            actions: (..., action_dim) action sequence.
                18-dim (from openarmx.usd recording):
                  [0:14] interleaved arm joints, [14:16] left hand fingers, [16:18] right hand fingers.
                14-dim: delta poses matching ActionsCfg order [right_arm(6), right_grip(1), left_arm(6), left_grip(1)]

        Returns:
            Dict of eef_name -> (..., 1) gripper values (binary: >=0 open, <0 close).
        """
        dim = actions.shape[-1]
        if dim == 18:
            # Passive recordings store the full ordered DOF target vector from
            # openarmx.usd. Resolve finger columns from the articulation order
            # instead of assuming a fixed layout.
            joint_names = list(self.scene["robot"].joint_names)
            left_ids = [i for i, name in enumerate(joint_names) if "openarmx_left_finger" in name]
            right_ids = [i for i, name in enumerate(joint_names) if "openarmx_right_finger" in name]
            if len(left_ids) != 2 or len(right_ids) != 2:
                raise RuntimeError(
                    "Expected two left and two right finger joints in openarmx.usd action order, "
                    f"got left={left_ids}, right={right_ids}, names={joint_names}"
                )
            left_finger = actions[..., left_ids].mean(dim=-1, keepdim=True)
            right_finger = actions[..., right_ids].mean(dim=-1, keepdim=True)
            # Convert to binary signals: finger_close < threshold -> -1 (close), else 1 (open)
            left_grip = torch.where(left_finger < self._GRIP_THRESHOLD, -1.0, 1.0)
            right_grip = torch.where(right_finger < self._GRIP_THRESHOLD, -1.0, 1.0)
        elif dim == 14:
            # Delta-pose actions: gripper is already a binary signal
            # Layout: right_arm[0:6], right_grip[6:7], left_arm[7:13], left_grip[13:14]
            right_grip = actions[..., self._rg]
            left_grip = actions[..., self._lg]
        else:
            raise ValueError(f"Unexpected action dimension {dim}, expected 14 or 18")
        return {"left": left_grip, "right": right_grip}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Return cube poses from scene state plus fixed tray pose.

        The tray is part of the environment USD (not a RigidObject), so its pose
        is not tracked by the scene state. We manually add it here so MimicGen
        subtasks with ``object_ref="tray"`` can still look up the target position.
        """
        if env_ids is None:
            env_ids = slice(None)

        object_pose_matrix = super().get_object_poses(env_ids=env_ids)

        tray_pos_w = torch.tensor(
            (-1.71578, -0.0800, 0.1797),
            device=self.device, dtype=torch.float32,
        )
        tray_pos_rel = tray_pos_w - self.scene.env_origins[env_ids]
        tray_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        # Expand to batch so matrix_from_quat returns (N, 3, 3)
        num_envs = tray_pos_rel.shape[0]
        tray_quat_batch = tray_quat.unsqueeze(0).expand(num_envs, -1)
        tray_rot_mat = PoseUtils.matrix_from_quat(tray_quat_batch)
        object_pose_matrix["tray"] = PoseUtils.make_pose(tray_pos_rel, tray_rot_mat)
        return object_pose_matrix

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return subtask completion signals matching SubTaskConfig keys."""
        if env_ids is None:
            env_ids = slice(None)
        return {
            "grasp": self.obs_buf["subtask_terms"]["grasp"][env_ids].bool(),
            "place": self.obs_buf["subtask_terms"]["place"][env_ids].bool(),
        }
