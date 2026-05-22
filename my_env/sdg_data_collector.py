"""
Isaac Sim Replicator-based visual SDG (Synthetic Data Generation) for OpenArmX + Cube + Tray.

Captures RGB images, depth, 2D/3D bounding boxes, instance segmentation, and camera poses
with automated scene randomization.

Usage:
  # GUI mode
  ./isaaclab.sh -p my_env/sdg_data_collector.py

  # Headless (server)
  ./isaaclab.sh -p my_env/sdg_data_collector.py --headless --num_frames 100

  # With custom output dir
  ./isaaclab.sh -p my_env/sdg_data_collector.py --output_dir ./my_sdg_output --num_frames 500
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
    description="SDG data collection for OpenArmX cube-tray scene."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to capture.")
parser.add_argument(
    "--output_dir", type=str, default="./sdg_output", help="Output directory for captured data."
)
parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480], help="Camera resolution (W H).")
parser.add_argument("--num_cameras", type=int, default=3, help="Number of cameras to use.")
parser.add_argument("--capture_interval", type=int, default=30, help="Simulation steps between captures.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random
import time

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import torch
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

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
    choose_front_object_position,
    create_detection_tray,
    create_dynamic_cube,
    find_first_valid_prim,
    get_prim_world_position,
    set_joint_positions,
)


def setup_scene():
    """Load environment, robot, and create cube + tray programmatically."""
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

    add_sublayer(stage, ENV_USD)
    add_sublayer(stage, ROBOT_USD)
    add_debug_lights(stage)

    robot_path = find_first_valid_prim(stage, ROBOT_PRIM_CANDIDATES)
    robot_pos = get_prim_world_position(stage, robot_path)
    print(f"[INFO] Robot world position: {robot_pos}")

    cube_start_pos = choose_front_object_position(robot_pos)
    tray_center = TRAY_CENTER

    cube_path = create_dynamic_cube(
        stage, CUBE_PATH, pos=cube_start_pos, size=CUBE_SIZE,
        height=CUBE_HEIGHT, mass=CUBE_MASS,
    )
    tray_path = create_detection_tray(
        stage, TRAY_PATH, center=tray_center, size_x=TRAY_SIZE_X, size_y=TRAY_SIZE_Y,
        base_thickness=TRAY_BASE_THICKNESS, wall_thickness=TRAY_WALL_THICKNESS,
        wall_height=TRAY_WALL_HEIGHT, corner_radius=TRAY_CORNER_RADIUS,
    )

    all_init_joints = {**LEFT_INIT_JOINTS, **RIGHT_INIT_JOINTS}
    set_joint_positions(stage, robot_path, all_init_joints)

    sim.reset()
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

    # Semantic labels for Replicator
    add_labels(stage.GetPrimAtPath(cube_path), labels=["green_cube"], instance_name="class")
    add_labels(stage.GetPrimAtPath(tray_path), labels=["detection_tray"], instance_name="class")

    print(f"[INFO] Scene ready. cube={cube_path}, tray={tray_path}, robot={robot_path}")
    return sim, stage, cube_path, tray_path, cube_start_pos, robot_pos


def setup_replicator(stage, output_dir: str, resolution: tuple, num_cameras: int):
    """Set up Replicator cameras, render products, and writers."""
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    cameras = []
    render_products = []

    for i in range(num_cameras):
        cam_path = f"/World/SDG_Camera_{i}"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        cam_prim.CreateAttribute("focalLength", Sdf.ValueTypeNames.Float).Set(24.0)
        cam_prim.CreateAttribute("focusDistance", Sdf.ValueTypeNames.Float).Set(400.0)
        cam_prim.CreateAttribute("fStop", Sdf.ValueTypeNames.Float).Set(0.0)
        cam_prim.CreateAttribute("clippingRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.01, 10000.0))
        UsdGeom.Xformable(cam_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam_prim).AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

        cameras.append(cam_prim)
        rp = rep.create.render_product(cam_path, resolution)
        render_products.append(rp)

    os.makedirs(output_dir, exist_ok=True)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=True,
        bounding_box_3d=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        distance_to_camera=True,
        normals=True,
        camera_params=True,
        frame_padding=6,
    )
    writer.attach(render_products)

    return cameras, render_products, writer


def randomize_camera(cam_prim, target_center, min_dist=0.4, max_dist=1.5):
    """Randomize a camera pose to look near the target center from a random angle."""
    theta = random.uniform(0, 2 * 3.14159)
    phi = random.uniform(0.3, 1.2)
    distance = random.uniform(min_dist, max_dist)

    cx, cy, cz = target_center
    cam_x = cx + distance * random.uniform(-1.5, 1.5)
    cam_y = cy + distance * random.uniform(-1.5, 1.5)
    cam_z = cz + distance * random.uniform(0.5, 2.0)

    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))

    look_at = Gf.Vec3d(
        cx + random.uniform(-0.15, 0.15),
        cy + random.uniform(-0.15, 0.15),
        cz + random.uniform(-0.05, 0.05),
    )
    direction = (look_at - Gf.Vec3d(cam_x, cam_y, cam_z)).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(direction, up).GetNormalized()
    up_corrected = Gf.Cross(right, direction).GetNormalized()
    rot_mat = Gf.Matrix3d()
    rot_mat.SetCol(0, right)
    rot_mat.SetCol(1, up_corrected)
    rot_mat.SetCol(2, -direction)
    rot_quat = rot_mat.ExtractRotation().GetQuaternion()
    xform.AddRotateQuatOp().Set(rot_quat)


def main():
    sim, stage, cube_path, tray_path, cube_start_pos, robot_pos = setup_scene()

    resolution = tuple(args_cli.resolution)
    cameras, render_products, writer = setup_replicator(
        stage, args_cli.output_dir, resolution, args_cli.num_cameras
    )

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0)
    timeline.set_end_time(1000000)
    timeline.set_looping(False)
    timeline.play()
    timeline.commit()
    simulation_app.update()

    tray_center = TRAY_CENTER
    wall_start = time.perf_counter()

    for frame_idx in range(args_cli.num_frames):
        # Randomize camera poses
        if frame_idx % 3 == 0 or frame_idx == 0:
            for cam in cameras:
                focus = random.choice([tray_center, cube_start_pos])
                randomize_camera(cam, focus)

        # Optional: randomize scene lighting or cube position on reset
        # For now, the cube bounces naturally with physics

        # Simulate several steps between captures for scene dynamics
        for _ in range(args_cli.capture_interval):
            sim.step()

        rep.orchestrator.step(delta_time=0.0, rt_subframes=4, pause_timeline=False)

        if frame_idx % 10 == 0:
            elapsed = time.perf_counter() - wall_start
            print(f"[SDG] Frame {frame_idx}/{args_cli.num_frames}  elapsed={elapsed:.1f}s")

    rep.orchestrator.wait_until_complete()
    wall_duration = time.perf_counter() - wall_start
    num_captures = args_cli.num_frames * args_cli.num_cameras
    print(f"[SDG] Done. {num_captures} captures in {wall_duration:.1f}s -> {num_captures / wall_duration:.1f} fps")
    print(f"[SDG] Output: {os.path.abspath(args_cli.output_dir)}")

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
