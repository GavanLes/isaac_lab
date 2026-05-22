"""Export tray.usd from the same create_detection_tray used in the demo."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom
from scene_utils import create_detection_tray
from scene_config import (
    TRAY_CENTER, TRAY_SIZE_X, TRAY_SIZE_Y,
    TRAY_BASE_THICKNESS, TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT, TRAY_CORNER_RADIUS,
)

stage = Usd.Stage.CreateNew(os.path.join(os.path.dirname(__file__), "tray.usd"))
UsdGeom.Xform.Define(stage, "/World")
create_detection_tray(
    stage, "/World/Detection_Tray",
    center=(0.0, 0.0, 0.0),  # exported at origin; RigidObjectCfg.init_state.pos places it
    size_x=TRAY_SIZE_X,
    size_y=TRAY_SIZE_Y,
    base_thickness=TRAY_BASE_THICKNESS,
    wall_thickness=TRAY_WALL_THICKNESS,
    wall_height=TRAY_WALL_HEIGHT,
    corner_radius=TRAY_CORNER_RADIUS,
)
stage.Save()
print("[OK] tray.usd created")
simulation_app.close()
