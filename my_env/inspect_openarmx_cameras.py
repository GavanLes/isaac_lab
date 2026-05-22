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
            print("Camera:", prim.GetPath(), "Type:", prim.GetTypeName())

    if not found:
        print("No UsdGeom.Camera prim found.")

    print("\n[Possible camera-like prim names]")
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if (
            "cam" in name
            or "camera" in name
            or "wrist" in name
            or "head" in name
            or "rgb" in name
        ):
            print(prim.GetPath(), prim.GetTypeName())

    print("\n[Done]")


if __name__ == "__main__":
    main()
    simulation_app.close()