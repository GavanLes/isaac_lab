import argparse
import random
import select
import sys
import termios
import tty

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


# ============================================================
# 1. 启动 Isaac Sim / Isaac Lab
# ============================================================

parser = argparse.ArgumentParser(
    description="Load environment, robot, a cube, and a tray target with reset detection."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 启动后再导入 ROS2 bridge extension，避免和 USD 加载冲突
import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)


# ============================================================
# 2. AppLauncher 之后再导入 Isaac / USD 相关模块
# ============================================================

import omni.usd
import isaaclab.sim as sim_utils

from scene_config import (
    CAMERA_EYE,
    CAMERA_TARGET,
    CUBE_HEIGHT,
    CUBE_MASS,
    CUBE_PATH,
    CUBE_SIZE,
    EE_PRIM_CANDIDATES,
    ENV_USD,
    LEFT_INIT_JOINTS,
    RESET_COOLDOWN_STEPS,
    RIGHT_INIT_JOINTS,
    ROBOT_PRIM_CANDIDATES,
    ROBOT_USD,
    TRAY_BASE_THICKNESS,
    TRAY_CENTER,
    TRAY_CORNER_RADIUS,
    TRAY_PATH,
    TRAY_SIZE_X,
    TRAY_SIZE_Y,
    TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS,
)
from scene_utils import (
    add_debug_lights,
    add_sublayer,
    check_prims,
    choose_front_object_position,
    create_detection_tray,
    create_dynamic_cube,
    find_first_valid_prim,
    get_prim_world_position,
    is_cube_in_tray,
    print_stage_prims,
    reset_cube,
    set_joint_positions,
)


# ============================================================
# 3. 主程序
# ============================================================

def main():
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
    print("[DEBUG] ENV_USD      =", ENV_USD, flush=True)
    print("[DEBUG] ROBOT_USD    =", ROBOT_USD, flush=True)
    print("[DEBUG] CUBE_PATH   =", CUBE_PATH, flush=True)
    print("[DEBUG] TRAY_PATH    =", TRAY_PATH, flush=True)
    print("=" * 80, flush=True)

    add_sublayer(stage, ENV_USD)
    add_sublayer(stage, ROBOT_USD)
    add_debug_lights(stage)

    robot_path = find_first_valid_prim(stage, ROBOT_PRIM_CANDIDATES)
    robot_pos = get_prim_world_position(stage, robot_path)
    print("[INFO] Robot world position:", robot_pos, flush=True)

    ee_path = find_first_valid_prim(stage, EE_PRIM_CANDIDATES)
    print("[INFO] End-effector path:", ee_path, flush=True)

    cube_start_pos = choose_front_object_position(robot_pos)

    def random_cube_pos():
        import random
        return [
            robot_pos[0] + 0.35,
            robot_pos[1] + 0.17,
            robot_pos[2] + 0.20,
        ]
    tray_center = TRAY_CENTER

    cube_path = create_dynamic_cube(
        stage,
        CUBE_PATH,
        pos=random_cube_pos(),
        size=CUBE_SIZE,
        height=CUBE_HEIGHT,
        mass=CUBE_MASS,
    )
    tray_path = create_detection_tray(
        stage,
        TRAY_PATH,
        center=tray_center,
        size_x=TRAY_SIZE_X,
        size_y=TRAY_SIZE_Y,
        base_thickness=TRAY_BASE_THICKNESS,
        wall_thickness=TRAY_WALL_THICKNESS,
        wall_height=TRAY_WALL_HEIGHT,
        corner_radius=TRAY_CORNER_RADIUS,
    )

    # 设置机械臂初始关节角度
    all_init_joints = {**LEFT_INIT_JOINTS, **RIGHT_INIT_JOINTS}
    set_joint_positions(stage, robot_path, all_init_joints)

    sim.reset()
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

    print("=" * 80, flush=True)
    print("[INFO] All USD files loaded.", flush=True)
    print(f"[INFO] environment:        {ENV_USD}", flush=True)
    print(f"[INFO] robot:              {ROBOT_USD}", flush=True)
    print(f"[INFO] robot path:         {robot_path}", flush=True)
    print(f"[INFO] cube path:      {cube_path}", flush=True)
    print(f"[INFO] cube start pos: {cube_start_pos}", flush=True)
    print(f"[INFO] tray path:          {tray_path}", flush=True)
    print("=" * 80, flush=True)

    check_prims(stage, robot_path, cube_path, tray_path)
    print_stage_prims(stage)

    # Non-blocking keyboard input for manual reset
    _old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def key_pressed():
        return select.select([sys.stdin], [], [], 0)[0] != []

    def read_key():
        return sys.stdin.read(1)

    reset_cooldown = 0
    while simulation_app.is_running():
        sim.step()

        if reset_cooldown > 0:
            reset_cooldown -= 1
            # Flush keyboard buffer during cooldown (ignore buffered 'R' presses)
            while key_pressed():
                read_key()
            continue

        # Manual reset on 'R' key
        if key_pressed() and read_key().lower() == "r":
            sim.reset()
            new_pos = random_cube_pos()
            reset_cube(cube_path, new_pos)
            print(f"[INFO] Manual reset (R key) → cube: ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})")
            sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
            reset_cooldown = RESET_COOLDOWN_STEPS
            continue

        if is_cube_in_tray(
            stage,
            cube_path,
            tray_center,
            TRAY_SIZE_X,
            TRAY_SIZE_Y,
            TRAY_WALL_HEIGHT,
            ee_path=ee_path,
        ):
            sim.reset()
            new_pos = random_cube_pos()
            reset_cube(cube_path, new_pos)
            print(f"[INFO] Cube reset to: ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})")
            sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
            reset_cooldown = RESET_COOLDOWN_STEPS

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_settings)
    simulation_app.close()


if __name__ == "__main__":
    main()
