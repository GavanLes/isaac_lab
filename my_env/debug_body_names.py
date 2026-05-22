"""Debug script: print all body names of the OpenArm BI robot articulation.

Run:
  ./isaaclab.sh -p my_env/debug_body_names.py
"""

import argparse
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Debug body names.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from openarm_cube_tray_mimic_env_cfg import OpenArmCubeTrayMimicEnvCfg
from openarm_cube_tray_mimic_env import OpenArmCubeTrayMimicEnv

import isaaclab_tasks  # noqa: F401


def main():
    env_cfg = OpenArmCubeTrayMimicEnvCfg()
    env_cfg.scene.num_envs = 1

    # Remove FrameTransformer to avoid the rigid-body error during debug
    env_cfg.scene.ee_frame = None
    env_cfg.observations.policy.eef_pos = None
    env_cfg.observations.policy.eef_quat = None
    env_cfg.observations.policy.gripper_pos = None
    env_cfg.observations.policy.actions = None
    env_cfg.observations.subtask_terms.grasp = None
    env_cfg.observations.subtask_terms.place = None

    env_cfg.terminations = None
    env_cfg.recorders = None
    # Disable IK/action validation during debug
    env_cfg.actions.right_arm_action = None
    env_cfg.actions.right_gripper_action = None
    env_cfg.actions.left_arm_action = None
    env_cfg.actions.left_gripper_action = None

    print("[DEBUG] Creating environment...")
    env = OpenArmCubeTrayMimicEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    print(f"\n[DEBUG] Robot body_names ({len(robot.body_names)} total):")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    print(f"\n[DEBUG] Robot joint_names ({len(robot.joint_names)} total):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    # Also print the default joint positions
    print(f"\n[DEBUG] Default joint pos shape: {robot.data.default_joint_pos.shape}")
    print(f"[DEBUG] Default joint pos: {robot.data.default_joint_pos[0]}")

    env.close()
    simulation_app.close()
    print("[DEBUG] Done.")


if __name__ == "__main__":
    main()
