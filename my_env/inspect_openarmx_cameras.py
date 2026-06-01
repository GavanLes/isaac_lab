import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# 必须先启动 Isaac Sim / Kit
parser = argparse.ArgumentParser(description="Inspect cameras in openarmx.usd")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 启动后再 import pxr
from pxr import Usd, UsdGeom


USD_PATH = Path("/home/huatec/isaac_lab/my_env/openarmx.usd")


def main():
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {USD_PATH}")

    print("=" * 80)
    print(f"USD: {USD_PATH}")
    print("=" * 80)

    print("\n[Camera prims]")
    found = False
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            found = True
            cam = UsdGeom.Camera(prim)
            path = prim.GetPath()
            print(f"\n  Camera: {path}")
            # Resolution
            try:
                res = cam.GetResolutionAttr().Get()
                print(f"    resolution: {res}")
            except Exception:
                print("    resolution: N/A")
            # Focal length
            try:
                fl = cam.GetFocalLengthAttr().Get()
                print(f"    focal_length: {fl}")
            except Exception:
                print("    focal_length: N/A")
            # Apertures
            try:
                ha = cam.GetHorizontalApertureAttr().Get()
                print(f"    horizontal_aperture: {ha}")
            except Exception:
                print("    horizontal_aperture: N/A")
            try:
                va = cam.GetVerticalApertureAttr().Get()
                print(f"    vertical_aperture: {va}")
            except Exception:
                print("    vertical_aperture: N/A")
            # Clipping range
            try:
                cr = cam.GetClippingRangeAttr().Get()
                print(f"    clipping_range: {cr}")
            except Exception:
                print("    clipping_range: N/A")

    if not found:
        print("No UsdGeom.Camera prim found.")

    print("\n[ros2_camera_helper config]")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "OmniGraphNode" and "ros2_camera_helper" in prim.GetName():
            print(f"\n  Node: {prim.GetPath()}")
            # Check all attributes
            for attr in prim.GetAttributes():
                try:
                    val = attr.Get()
                    if val is not None and str(val) not in ("", "[]", "None"):
                        print(f"    {attr.GetName()}: {val}")
                except Exception:
                    pass

    print("\n[Done]")


if __name__ == "__main__":
    main()
    simulation_app.close()