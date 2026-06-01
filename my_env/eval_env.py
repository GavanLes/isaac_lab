"""Eval script using EXACT same env as MimicGen datagen.

Usage: ./isaaclab.sh -p my_env/eval_env.py

Builds the identical ManagerBasedRLMimicEnv. The robot is controlled by
the Action Graph subscribing to /openarmx_isaac_joint_commands (published
by ros2 launch openarmx_command_to_joint_state.launch.py).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Eval in datagen env.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable ROS2 bridge before loading stage
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
print("[INFO] ROS2 Bridge enabled", flush=True)

import torch
import openarm_cube_tray_mimic_env as _env_mod
import openarm_cube_tray_mimic_env_cfg as _cfg_mod


def main():
    # ---- Build env with fixed cube position (no randomization) ----
    # Monkey-patch random to always return 0 (no cube position noise)
    import random as _real_random
    _noop_random = lambda a, b: 0.0
    _randomize_fn = _cfg_mod._randomize_cube_pose
    def _patched_randomize(env, env_ids, asset_cfg=None):
        import random
        random.uniform = lambda a, b: 0.0
        _randomize_fn(env, env_ids, asset_cfg)
        random.uniform = _real_random.uniform
    _cfg_mod._randomize_cube_pose = _patched_randomize

    env_cfg = _cfg_mod.OpenArmCubeTrayMimicEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations = None
    env = _env_mod.OpenArmCubeTrayMimicEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    dof_names = list(robot.joint_names)
    print(f"[INFO] {len(dof_names)} DOFs", flush=True)
    print(f"[INFO] PD: arm=100000/2000, gripper=50000/500", flush=True)
    print(f"[INFO] Robot spawned via ArticulationCfg + ImplicitActuatorCfg", flush=True)
    print("=" * 60, flush=True)
    print("[INFO] Action Graph controls robot via /openarmx_isaac_joint_commands", flush=True)
    print("[INFO] Run: ros2 launch isaacsim openarmx_command_to_joint_state.launch.py", flush=True)
    print("=" * 60, flush=True)

    step = 0
    while simulation_app.is_running():
        env.sim.step()
        step += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
