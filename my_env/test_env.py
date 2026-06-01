"""
Test script that opens the SAME environment as data generation (training).

Usage:
  ./isaaclab.sh -p my_env/test_env.py [--save_images]

Camera images are saved to /tmp/test_env_*.png on the first step for
direct comparison with training data videos.

Controls:
  R — reset
  Q — quit
"""

import argparse
import os
import select
import sys
import termios
import tty

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(
    description="Open training environment for visual inspection."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--save_images", action="store_true",
                    help="Save first camera images to /tmp/test_env_*.png")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable ROS2 bridge so openarmx.usd cameras publish ROS2 topics
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

import numpy as np
import torch

import openarm_cube_tray_mimic_env as _env_mod
import openarm_cube_tray_mimic_env_cfg as _cfg_mod


def main():
    env_cfg = _cfg_mod.OpenArmCubeTrayMimicEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.env_name = "Isaac-OpenArm-Cube-Tray-Mimic-v0"
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.datagen_config.generation_num_trials = 1

    # Keep recorder cfg but don't export (avoids issues with None recorders)
    env_cfg.recorders = _cfg_mod.OpenArmXRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = "/tmp"

    env = _env_mod.OpenArmCubeTrayMimicEnv(cfg=env_cfg)

    env.reset()
    # Settle
    for _ in range(10):
        env.step(torch.zeros(1, 14, device=env.device))

    print("=" * 60)
    print("[INFO] Environment ready. Same config as training.")
    print(f"[INFO] Cameras: left_hand_cam, right_hand_cam, body_cam")
    print(f"[INFO] Image size: 240x240")
    print(f"[INFO] Robot pos: {_cfg_mod._ROBOT_POS}")
    print(f"[INFO] PhysX: cpu, dt=0.01, render_interval=2")
    print("[INFO] Press 'R' to reset  |  'Q' to quit")
    print("=" * 60)

    _old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def key_pressed():
        return select.select([sys.stdin], [], [], 0)[0] != []

    def read_key():
        return sys.stdin.read(1)

    images_saved = False
    step_count = 0
    while simulation_app.is_running():
        # Hold position, grippers open
        action = torch.zeros(1, 14, device=env.device)
        action[:, 6] = 1.0   # right gripper open
        action[:, 13] = 1.0  # left gripper open

        obs, _, _, _, _ = env.step(action)
        step_count += 1

        # Print obs structure on first step
        if step_count == 1:
            print(f"[DEBUG] Observation keys: {sorted(obs.keys())}")
            for k, v in obs.items():
                if isinstance(v, dict):
                    print(f"[DEBUG]   {k}/ — keys: {sorted(v.keys())}")
                elif isinstance(v, torch.Tensor):
                    print(f"[DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"[DEBUG]   {k}: type={type(v).__name__}")

        # Save first frame images for comparison
        if args_cli.save_images and not images_saved and step_count > 2:
            from PIL import Image

            # Find camera images in obs dict
            cam_data = obs.get("policy", obs)  # might be nested under "policy"
            for cam_name in ["left_hand_cam", "right_hand_cam", "body_cam"]:
                img_tensor = None
                if cam_name in obs:
                    img_tensor = obs[cam_name]
                elif cam_name in cam_data:
                    img_tensor = cam_data[cam_name]

                if img_tensor is not None:
                    img = img_tensor[0].cpu().numpy()
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    out_path = f"/tmp/test_env_{cam_name}.png"
                    Image.fromarray(img).save(out_path)
                    print(f"[INFO] Saved {out_path}  shape={img.shape}  range=[{img.min()},{img.max()}]")
                else:
                    print(f"[WARN] Camera '{cam_name}' not found in obs. Available keys: {sorted(obs.keys())}")
            images_saved = True

        if key_pressed():
            key = read_key().lower()
            if key == "q":
                print("[INFO] Quit.")
                break
            elif key == "r":
                env.reset()
                for _ in range(10):
                    env.step(torch.zeros(1, 14, device=env.device))
                print("[INFO] Reset.")
                step_count = 0
                images_saved = False

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_settings)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
