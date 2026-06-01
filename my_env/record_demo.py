"""
Passive recorder for the OpenArmX Cube-to-Tray task.

Loads demo.py's scene (environment, robot, cube, tray, lights, camera) and
allows keyboard teleop while recording EEF poses, object states, gripper
positions, and subtask signals to an HDF5 dataset.

The output HDF5 follows MimicGen conventions and can be used as a source
demonstration for MimicGen data generation.

Usage:
  # Keyboard teleop + GUI
  ./isaaclab.sh -p my_env/record_demo.py --output ./datasets/cube_tray_source.hdf5

  # Headless with scripted controller
  ./isaaclab.sh -p my_env/record_demo.py --headless --scripted

Controls:
  W/S        move EE +X/-X
  A/D        move EE +Y/-Y
  Q/E        move EE +Z/-Z
  Z/X        rotate EE +roll/-roll
  T/G        rotate EE +pitch/-pitch
  C/V        rotate EE +yaw/-yaw
  K          toggle gripper (open/close)
  L          reset pose
  ENTER      finish episode, save, and start next episode
  ESC/close  quit recording
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(
    description="Record source demo for OpenArmX Cube-to-Tray."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--output", type=str, default="./datasets/cube_tray_source.hdf5",
    help="Output HDF5 dataset path.",
)
parser.add_argument(
    "--episode_length_s", type=float, default=30.0,
    help="Max episode length in seconds.",
)
parser.add_argument(
    "--scripted", action="store_true",
    help="Use a scripted controller instead of keyboard teleop (for headless CI).",
)
parser.add_argument(
    "--passive", action="store_true",
    help="Passive recording: do not apply IK/joint targets. "
         "Use this when an external controller (ROS2/VR/Action Graph) drives the robot.",
)
parser.add_argument(
    "--episodes", type=int, default=0,
    help="Number of episodes to record (0 = infinite, press ENTER to go to next).",
)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _enable_ros2_bridge_before_loading_stage(simulation_app):
    """Enable Isaac Sim ROS2 Bridge before opening USD or creating env.

    The USD stage contains ROS2 Action Graph nodes (ROS2PublishJointState,
    ROS2Context, etc.) that require the isaacsim.ros2.bridge extension to be
    loaded before the stage is opened.
    """
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()

    candidate_exts = [
        "isaacsim.ros2.bridge",       # Isaac Sim 4.5/5.x
        "omni.isaac.ros2_bridge",     # older Isaac Sim fallback
    ]

    last_error = None
    for ext_name in candidate_exts:
        try:
            if not ext_manager.is_extension_enabled(ext_name):
                ext_manager.set_extension_enabled_immediate(ext_name, True)

            for _ in range(10):
                simulation_app.update()

            if ext_manager.is_extension_enabled(ext_name):
                print(f"[INFO] ROS2 Bridge enabled: {ext_name}")
                return ext_name
        except Exception as e:
            last_error = e
            print(f"[WARN] Failed to enable ROS2 Bridge extension {ext_name}: {e}")

    raise RuntimeError(
        "Failed to enable ROS2 Bridge. "
        "The USD contains ROS2 Action Graph nodes such as "
        "'isaacsim.ros2.bridge.ROS2PublishJointState', but the bridge extension "
        f"could not be enabled. Last error: {last_error}"
    )


_enable_ros2_bridge_before_loading_stage(simulation_app)

import numpy as np
import torch

import omni.usd
from pxr import Gf
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController

from scene_config import (
    CAMERA_EYE, CAMERA_TARGET, CUBE_HEIGHT, CUBE_MASS, CUBE_PATH,
    CUBE_SIZE, EE_PRIM_CANDIDATES, ENV_USD, LEFT_INIT_JOINTS,
    RIGHT_INIT_JOINTS, ROBOT_PRIM_CANDIDATES, ROBOT_USD,
    TRAY_BASE_THICKNESS, TRAY_CENTER, TRAY_PRIM_CANDIDATES,
    TRAY_SIZE_X, TRAY_SIZE_Y, TRAY_WALL_HEIGHT,
)

from scene_utils import (
    add_debug_lights, add_sublayer, choose_front_object_position,
    create_dynamic_cube, find_first_valid_prim,
    get_prim_world_position, reset_physx_body, set_joint_positions,
)


def main():
    # ------------------------------------------------------------------
    # Build scene (from demo.py)
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device="cpu",
        physx=sim_utils.PhysxCfg(
            solve_articulation_contact_last=True,
            enable_ccd=True,
            min_position_iteration_count=8,
            max_position_iteration_count=64,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=32,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

    print("=" * 80, flush=True)
    add_sublayer(stage, ENV_USD)
    add_sublayer(stage, ROBOT_USD)
    add_debug_lights(stage)

    robot_path = find_first_valid_prim(stage, ROBOT_PRIM_CANDIDATES)
    robot_pos = get_prim_world_position(stage, robot_path)
    print(f"[INFO] Robot world position: {robot_pos}", flush=True)

    ee_path = find_first_valid_prim(stage, EE_PRIM_CANDIDATES)
    print(f"[INFO] End-effector path: {ee_path}", flush=True)

    cube_start_pos = choose_front_object_position(robot_pos)

    # Read tray position dynamically from the environment USD
    tray_path = find_first_valid_prim(stage, TRAY_PRIM_CANDIDATES)
    tray_center = get_prim_world_position(stage, tray_path)
    print(f"[INFO] Tray prim path: {tray_path}", flush=True)
    print(f"[INFO] Tray world position: {tray_center}", flush=True)

    cube_path = create_dynamic_cube(
        stage, CUBE_PATH,
        pos=cube_start_pos, size=CUBE_SIZE, height=CUBE_HEIGHT, mass=CUBE_MASS,
    )

    all_init_joints = {**LEFT_INIT_JOINTS, **RIGHT_INIT_JOINTS}
    set_joint_positions(stage, robot_path, all_init_joints)

    sim.reset()
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
    print("=" * 80, flush=True)

    # ------------------------------------------------------------------
    # Physics views for state queries & control
    # ------------------------------------------------------------------
    from isaacsim.core.simulation_manager import SimulationManager

    physics_view = SimulationManager.get_physics_sim_view()
    _articulation_view = physics_view.create_articulation_view(robot_path)
    _cube_view = physics_view.create_rigid_body_view(cube_path)

    # Map link/joint names → indices
    link_names = list(_articulation_view.shared_metatype.link_names)
    dof_names = list(_articulation_view.shared_metatype.dof_names)
    print(f"[INFO] Articulation has {len(link_names)} links, {len(dof_names)} DOFs")
    print(f"[INFO] Link names sample: {link_names[:5]}...")
    print(f"[INFO] DOF names sample: {dof_names[:5]}...")

    # Directly set joint positions to match ROS2 initial pose (bypass PD controller)
    init_pos = torch.zeros(1, len(dof_names), dtype=torch.float32)
    for i, name in enumerate(dof_names):
        if name in all_init_joints:
            init_pos[0, i] = float(all_init_joints[name])
    init_indices = torch.tensor([0], dtype=torch.int32)
    _articulation_view.set_dof_positions(init_pos, indices=init_indices)
    print(f"[INFO] Joint positions set directly: {len(dof_names)} DOFs")

    # Find left_hand link index
    right_hand_idx = None
    left_hand_idx = None
    for i, name in enumerate(link_names):
        if "right_hand" in name and right_hand_idx is None:
            right_hand_idx = i
        if "left_hand" in name and left_hand_idx is None:
            left_hand_idx = i
    if left_hand_idx is None:
        print("[ERROR] Could not find left_hand link. Link names:")
        for n in link_names:
            print(f"  {n}")
        simulation_app.close()
        return
    print(f"[INFO] right_hand link idx: {right_hand_idx}  ({link_names[right_hand_idx] if right_hand_idx is not None else 'N/A'})")
    print(f"[INFO] left_hand link idx:  {left_hand_idx}  ({link_names[left_hand_idx]})")

    # Find left_finger / left_arm DOF indices
    right_finger_indices = []
    left_finger_indices = []
    right_arm_indices = []
    left_arm_indices = []
    for i, name in enumerate(dof_names):
        if "right_finger" in name:
            right_finger_indices.append(i)
        elif "left_finger" in name:
            left_finger_indices.append(i)
        elif "right_joint" in name:
            right_arm_indices.append(i)
        elif "left_joint" in name:
            left_arm_indices.append(i)
    print(f"[INFO] left_finger DOF idx: {left_finger_indices}")
    print(f"[INFO] left_arm DOF idx: {left_arm_indices}")

    # ------------------------------------------------------------------
    # IK Controller
    # ------------------------------------------------------------------
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls",
    )
    ik_ctrl = DifferentialIKController(ik_cfg, num_envs=1, device="cpu")

    # ------------------------------------------------------------------
    # Keyboard teleop (skipped in headless or passive mode)
    # ------------------------------------------------------------------
    teleop = None
    passive_mode = args_cli.passive
    _enter_flag = {"pressed": False}
    if passive_mode:
        print("[INFO] Passive recording mode — external controller (ROS2/VR) drives the robot.")
        print("[INFO] Press ENTER to save & start next episode. Close window to quit.\n")

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        _passive_kb = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
        _passive_kb.reset()
        _passive_kb.add_callback("ENTER", lambda: _enter_flag.update(pressed=True))
    elif not args_cli.headless:
        os.environ.setdefault("__record_enter_pressed", "0")
        os.environ.setdefault("__record_gripper_closed", "0")

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.1))
        teleop.reset()

        def _on_enter():
            os.environ["__record_enter_pressed"] = "1"
        def _on_reset():
            os.environ["__record_reset"] = "1"
        teleop.add_callback("ENTER", _on_enter)
        teleop.add_callback("L", _on_reset)

        print("\n[TELEOP] Controls:")
        print("  W/S = pos +/-X    A/D = pos +/-Y    Q/E = pos +/-Z")
        print("  Z/X = rot roll    T/G = rot pitch   C/V = rot yaw")
        print("  K = toggle gripper  L = reset pose")
        print("  ENTER = save & next episode\n")
    else:
        print("[INFO] Headless mode. Robot will hold initial pose.", flush=True)

    print("[INFO] Start controlling the robot. Press ENTER to save & go to next episode.")

    # ------------------------------------------------------------------
    # HDF5 handler (created once for all episodes)
    # ------------------------------------------------------------------
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

    output_dir = os.path.dirname(args_cli.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    handler = HDF5DatasetFileHandler()
    handler.create(args_cli.output, "openarm_cube_tray")

    max_steps = int(args_cli.episode_length_s / sim_cfg.dt)

    robot_pos_t = torch.tensor(robot_pos, dtype=torch.float32)
    tray_center_t = torch.tensor(tray_center, dtype=torch.float32)
    # Dynamic wall-top threshold (base top + wall height + tolerance)
    tray_wall_top_z = tray_center[2] + TRAY_BASE_THICKNESS * 0.5 + TRAY_WALL_HEIGHT + 0.005
    left_finger_idx_t = torch.tensor(left_finger_indices, dtype=torch.long)
    left_arm_idx_t = torch.tensor(left_arm_indices, dtype=torch.long)

    # ------------------------------------------------------------------
    # Episode loop (infinite if episodes=0, press ENTER → next episode)
    # ------------------------------------------------------------------
    ep = 0
    max_episodes = args_cli.episodes

    while simulation_app.is_running():
        if max_episodes > 0 and ep >= max_episodes:
            break

        # --- reset scene for new episode ---
        if ep > 0:
            new_cube_pos = choose_front_object_position(robot_pos)
            print(f"\n[INFO] Resetting cube to {new_cube_pos}")
            reset_physx_body(cube_path, new_cube_pos)
            set_joint_positions(stage, robot_path, all_init_joints)
            sim.reset()
            sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
            _enter_flag["pressed"] = False
            if teleop is not None:
                teleop.reset()
                os.environ["__record_enter_pressed"] = "0"
                os.environ["__record_reset"] = "0"
            # Re-obtain physics_view and recreate views after sim.reset()
            physics_view = SimulationManager.get_physics_sim_view()
            _articulation_view = physics_view.create_articulation_view(robot_path)
            _cube_view = physics_view.create_rigid_body_view(cube_path)
            # Directly set joint positions to match ROS2 initial pose
            dof_names_reset = list(_articulation_view.shared_metatype.dof_names)
            init_pos_reset = torch.zeros(1, len(dof_names_reset), dtype=torch.float32)
            for i, name in enumerate(dof_names_reset):
                if name in all_init_joints:
                    init_pos_reset[0, i] = float(all_init_joints[name])
            idx_tensor = torch.tensor([0], dtype=torch.int32)
            _articulation_view.set_dof_positions(init_pos_reset, indices=idx_tensor)
            # Also sync position targets so the PD controller doesn't fight the reset
            _articulation_view.set_dof_position_targets(init_pos_reset, indices=idx_tensor)
            # Let PD controller settle
            for _ in range(30):
                sim.step()

        total_str = f"/{max_episodes}" if max_episodes > 0 else ""
        print(f"\n{'='*60}")
        print(f"[EPISODE {ep + 1}{total_str}]")
        print(f"{'='*60}\n")

        # --- Recording buffers ---
        actions_list = []
        obs_eef_pos_list = []
        obs_eef_quat_list = []
        obs_gripper_pos_list = []
        obs_cube_pos_list = []
        subtask_grasp_list = []
        subtask_place_list = []

        step_count = 0
        episode_done = False

        # --- Main loop ---
        while simulation_app.is_running() and not episode_done and step_count < max_steps:
            # --- read current state from physics ---
            link_transforms = _articulation_view.get_link_transforms()  # (1, N_links, 7)
            link_t = link_transforms[0]  # (N_links, 7)
            dof_pos = _articulation_view.get_dof_positions()  # (1, N_dof)
            dof_pos_t = dof_pos[0]  # (N_dof,)
            cube_transform = _cube_view.get_transforms()  # (1, 7)

            # Left hand pose in world frame
            lhand_pos = link_t[left_hand_idx, :3]
            lhand_quat = link_t[left_hand_idx, 3:7]  # xyzw
            lhand_quat_wxyz = torch.cat([lhand_quat[3:4], lhand_quat[:3]])

            cube_pos = cube_transform[0, :3]

            # Gripper position (mean of both fingers)
            gripper_val = dof_pos_t[left_finger_indices].mean()

            # --- subtask signals ---
            # grasp: gripper partially closed (cube blocks full closure) + hand near cube
            # Actual grasp: dist≈0.09, grip≈0.021 (can't fully close due to cube)
            hand_cube_dist = torch.linalg.norm(lhand_pos - cube_pos, dim=-1)
            grasped = (hand_cube_dist < 0.12) and (gripper_val < 0.03)

            # place: cube inside tray XY bounds + above base + below walls
            dx = abs(cube_pos[0].item() - tray_center_t[0].item())
            dy = abs(cube_pos[1].item() - tray_center_t[1].item())
            inside_xy = dx < TRAY_SIZE_X * 0.5 and dy < TRAY_SIZE_Y * 0.5
            above_base = cube_pos[2].item() > tray_center_t[2].item() + 0.002
            below_top = cube_pos[2].item() < tray_wall_top_z
            in_tray = inside_xy and above_base and below_top

            # --- record observations ---
            obs_eef_pos_list.append(lhand_pos.clone())
            obs_eef_quat_list.append(lhand_quat_wxyz.clone())
            obs_gripper_pos_list.append(gripper_val.clone().unsqueeze(0))
            obs_cube_pos_list.append(cube_pos.clone())
            subtask_grasp_list.append(torch.tensor(grasped, dtype=torch.bool))
            subtask_place_list.append(torch.tensor(in_tray, dtype=torch.bool))

            # --- compute action ---
            if teleop is not None:
                teleop_state = teleop.advance()
                # teleop_state: [x, y, z, rx, ry, rz, gripper]
                delta_pos = teleop_state[:3].detach().clone() * 0.05
                delta_rot_aa = teleop_state[3:6].detach().clone() * 0.1
                gripper_cmd = teleop_state[6].detach().clone()  # +1 open, -1 close

                ik_command = torch.cat([delta_pos, delta_rot_aa]).unsqueeze(0)

                ee_pos_batch = lhand_pos.unsqueeze(0)
                ee_quat_batch = lhand_quat_wxyz.unsqueeze(0)
                jacobians = _articulation_view.get_jacobians()
                jac_lhand_full = jacobians[0, left_hand_idx]
                jac_lhand = jac_lhand_full[:, left_arm_indices]
                left_arm_pos = dof_pos_t[left_arm_indices].unsqueeze(0)

                ik_ctrl.set_command(ik_command, ee_pos_batch, ee_quat_batch)
                target_left_arm = ik_ctrl.compute(
                    ee_pos_batch, ee_quat_batch,
                    jac_lhand.unsqueeze(0), left_arm_pos,
                )

                target_full = dof_pos_t.clone().unsqueeze(0)
                target_full[0, left_arm_indices] = target_left_arm[0]
                if len(left_finger_indices) == 2:
                    if gripper_cmd < 0:
                        finger_target = torch.zeros(2)
                    else:
                        finger_target = torch.full((2,), 0.04)
                    for idx_i, dof_idx in enumerate(left_finger_indices):
                        target_full[0, dof_idx] = finger_target[idx_i]
                _articulation_view.set_dof_position_targets(
                    target_full, torch.tensor([0], dtype=torch.int32),
                )

                action = torch.zeros(14, dtype=torch.float32)
                action[6] = 1.0
                action[7:13] = ik_command[0]
                action[13] = gripper_cmd
                actions_list.append(action)

                if os.environ.get("__record_enter_pressed") == "1":
                    episode_done = True
                if os.environ.get("__record_reset") == "1":
                    set_joint_positions(stage, robot_path, all_init_joints)
                    sim.reset()
                    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
                    os.environ["__record_reset"] = "0"
                    # Re-obtain physics views and sync PD targets after reset
                    physics_view = SimulationManager.get_physics_sim_view()
                    _articulation_view = physics_view.create_articulation_view(robot_path)
                    _cube_view = physics_view.create_rigid_body_view(cube_path)
                    init_pos = torch.zeros(1, len(dof_names), dtype=torch.float32)
                    for i, name in enumerate(dof_names):
                        if name in all_init_joints:
                            init_pos[0, i] = float(all_init_joints[name])
                    idx_t = torch.tensor([0], dtype=torch.int32)
                    _articulation_view.set_dof_positions(init_pos, indices=idx_t)
                    _articulation_view.set_dof_position_targets(init_pos, indices=idx_t)
                    for _ in range(30):
                        sim.step()
                    print("[INFO] Pose reset.")
            elif passive_mode:
                # Passive: record joint position targets set by Action Graph / ROS2
                # These are the actual VR → IK → joint targets the robot is following
                dof_targets = _articulation_view.get_dof_position_targets()  # (1, N_dof)
                action = dof_targets[0].detach().clone()  # (N_dof,)
                actions_list.append(action)

                if _enter_flag["pressed"]:
                    episode_done = True
            else:
                # Headless — zero action (hold position)
                action = torch.zeros(14, dtype=torch.float32)
                action[6] = 1.0   # right gripper open (idle)
                action[13] = 1.0  # left gripper open
                actions_list.append(action)

            # Step physics
            sim.step()
            step_count += 1

            if step_count % 10 == 0:
                dz = lhand_pos[2].item() - cube_pos[2].item()
                print(f"[STEP {step_count}] EEF z={lhand_pos[2].item():.4f}  cube z={cube_pos[2].item():.4f}  "
                      f"dz={dz:.4f}  dist={hand_cube_dist.item():.4f}  grip={gripper_val.item():.4f}  "
                      f"grasp={grasped}  in_tray={in_tray}")

        # --- Export episode to HDF5 ---
        print(f"\n[INFO] Episode {ep + 1} finished. {step_count} steps collected.")

        episode = EpisodeData()
        for a in actions_list:
            episode.add("actions", a)
        for p in obs_eef_pos_list:
            episode.add("obs/eef_pos", p)
        for q in obs_eef_quat_list:
            episode.add("obs/eef_quat", q)
        for g in obs_gripper_pos_list:
            episode.add("obs/gripper_pos", g)
        for c in obs_cube_pos_list:
            episode.add("obs/cube_pos", c)
        for gr in subtask_grasp_list:
            episode.add("obs/subtask_terms/grasp", gr)
        for pl in subtask_place_list:
            episode.add("obs/subtask_terms/place", pl)
        episode.pre_export()
        handler.write_episode(episode, demo_id=ep)
        print(f"[INFO] Episode {ep + 1} saved as demo_{ep}. Data keys: {list(episode.data.keys())}")

        ep += 1

    # --- finalize ---
    handler.close()
    print(f"\n[INFO] All {ep} episodes saved to: {args_cli.output}")
    simulation_app.close()


if __name__ == "__main__":
    main()
