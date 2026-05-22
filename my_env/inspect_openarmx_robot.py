import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Inspect OpenArmX USD robot structure.")
parser.add_argument("--usd", type=Path, default=Path("/home/huatec/isaac_lab/my_env/openarmx.usd"))
args_cli = parser.parse_args()

from pxr import Usd, UsdGeom, UsdPhysics


def _target_paths(attr):
    if not attr:
        return []
    targets = attr.GetTargets()
    return [str(path) for path in targets]


def _print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _joint_type(prim):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "RevoluteJoint"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "PrismaticJoint"
    if prim.IsA(UsdPhysics.FixedJoint):
        return "FixedJoint"
    if prim.IsA(UsdPhysics.SphericalJoint):
        return "SphericalJoint"
    if prim.IsA(UsdPhysics.Joint):
        return "Joint"
    return prim.GetTypeName()


def _joint_limits(prim):
    low = prim.GetAttribute("physics:lowerLimit")
    high = prim.GetAttribute("physics:upperLimit")
    low_value = low.Get() if low else None
    high_value = high.Get() if high else None
    if low_value is None and high_value is None:
        return ""
    return f" limits=[{low_value}, {high_value}]"


def _drive_summary(prim):
    schemas = prim.GetAppliedSchemas()
    drive_schemas = [schema for schema in schemas if schema.startswith("PhysicsDriveAPI")]
    if not drive_schemas:
        return ""
    return " drives=" + ",".join(drive_schemas)


def main():
    stage = Usd.Stage.Open(str(args_cli.usd))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args_cli.usd}")

    print("=" * 80)
    print(f"USD: {args_cli.usd}")
    print("=" * 80)

    articulation_roots = []
    rigid_bodies = []
    joints = []
    cameras = []
    interesting = []

    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
        if prim.IsA(UsdGeom.Camera):
            cameras.append(prim)
        if any(token in name for token in ("left", "right", "hand", "finger", "link", "camera", "cam")):
            interesting.append(prim)

    _print_section("Articulation roots")
    if articulation_roots:
        for prim in articulation_roots:
            print(f"{prim.GetPath()}  type={prim.GetTypeName()}")
    else:
        print("No prim has UsdPhysics.ArticulationRootAPI.")

    _print_section("Rigid bodies / candidate body names")
    if rigid_bodies:
        for prim in rigid_bodies:
            print(f"{prim.GetPath()}  name={prim.GetName()}  type={prim.GetTypeName()}")
    else:
        print("No prim has UsdPhysics.RigidBodyAPI.")

    _print_section("Physics joints / candidate joint names")
    if joints:
        for prim in joints:
            joint = UsdPhysics.Joint(prim)
            body0 = _target_paths(joint.GetBody0Rel())
            body1 = _target_paths(joint.GetBody1Rel())
            print(
                f"name={prim.GetName()}  path={prim.GetPath()}  type={_joint_type(prim)}"
                f"{_joint_limits(prim)}{_drive_summary(prim)}"
            )
            print(f"  body0={body0}")
            print(f"  body1={body1}")
    else:
        print("No UsdPhysics.Joint prim found.")

    _print_section("Left arm likely joints")
    for prim in joints:
        text = str(prim.GetPath()).lower()
        if "left" in text:
            print(prim.GetName())

    _print_section("Right arm likely joints")
    for prim in joints:
        text = str(prim.GetPath()).lower()
        if "right" in text:
            print(prim.GetName())

    _print_section("Hand / finger / camera / link prims")
    for prim in interesting:
        print(f"{prim.GetPath()}  name={prim.GetName()}  type={prim.GetTypeName()}  schemas={prim.GetAppliedSchemas()}")

    _print_section("Cameras")
    if cameras:
        for prim in cameras:
            print(f"{prim.GetPath()}  name={prim.GetName()}")
    else:
        print("No UsdGeom.Camera prim found.")

    print("\n[Done]")


if __name__ == "__main__":
    main()
