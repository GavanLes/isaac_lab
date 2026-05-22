"""Inspect the spawned cube and compare with create_dynamic_cube() expectations.

Run this INSIDE Isaac Lab simulation (after scene creation).
Usage: Call inspect_cube(stage) from a script that has access to the simulation stage.
"""
from pxr import UsdPhysics, UsdGeom, UsdShade, PhysxSchema


def inspect_cube(stage, cube_path="/World/envs/env_0/Cube"):
    """Print all physics properties of a spawned cube for comparison."""
    prim = stage.GetPrimAtPath(cube_path)
    if not prim.IsValid():
        print(f"ERROR: Prim not found at {cube_path}")
        # Try to find any Cube prim
        for p in stage.Traverse():
            if "Cube" in str(p.GetPath()):
                print(f"  Found: {p.GetPath()} type={p.GetTypeName()}")
        return

    print(f"=== Cube at {cube_path} ===")
    print(f"  Type: {prim.GetTypeName()}")
    print(f"  Has RigidBodyAPI: {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
    print(f"  Has CollisionAPI: {prim.HasAPI(UsdPhysics.CollisionAPI)}")
    print(f"  Has MassAPI: {prim.HasAPI(UsdPhysics.MassAPI)}")
    print(f"  Has PhysicsMaterialAPI: {prim.HasAPI(UsdPhysics.MaterialAPI)}")

    # Children
    children = list(prim.GetChildren())
    print(f"  Children ({len(children)}):")
    for c in children:
        print(f"    {c.GetPath()} type={c.GetTypeName()}")
        apis = [api for api in ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI", "PhysicsMaterialAPI"]
                if c.HasAPI(getattr(UsdPhysics, api.split("Physics")[-1], None) or getattr(UsdPhysics, api, None))]
        if apis:
            print(f"      APIs: {apis}")

    # Cube geometry
    cube = UsdGeom.Cube(prim)
    if cube:
        try:
            print(f"  Size: {cube.GetSizeAttr().Get()}")
        except Exception:
            print("  Size: N/A (not a Cube?)")

    # Xform ops
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    print(f"  XformOps ({len(ops)}):")
    for op in ops:
        try:
            print(f"    {op.GetName()} = {op.Get()}")
        except Exception:
            print(f"    {op.GetName()} = <error>")

    # Rigid body
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(prim)
        print(f"  rigidBodyEnabled: {rb.GetRigidBodyEnabledAttr().Get()}")
        print(f"  kinematicEnabled: {rb.GetKinematicEnabledAttr().Get()}")
        try:
            print(f"  startsAsleep: {rb.GetStartsAsleepAttr().Get()}")
        except Exception:
            print("  startsAsleep: N/A")

    # Mass
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass = UsdPhysics.MassAPI(prim)
        try:
            print(f"  mass: {mass.GetMassAttr().Get()}")
        except Exception:
            print("  mass: N/A")
        try:
            print(f"  density: {mass.GetDensityAttr().Get()}")
        except Exception:
            print("  density: N/A")
        try:
            print(f"  diagonalInertia: {mass.GetDiagonalInertiaAttr().Get()}")
        except Exception:
            print("  diagonalInertia: N/A (auto-computed)")

    # Collision
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        col = UsdPhysics.CollisionAPI(prim)
        try:
            print(f"  collisionEnabled: {col.GetCollisionEnabledAttr().Get()}")
        except Exception:
            print("  collisionEnabled: N/A")

    # Material binding
    mb = UsdShade.MaterialBindingAPI(prim)
    mat, purpose = mb.ComputeBoundMaterial()
    if mat:
        print(f"  Bound material: {mat.GetPath()} (purpose={purpose})")
        mat_prim = mat.GetPrim()
        if mat_prim.HasAPI(UsdPhysics.MaterialAPI):
            pm = UsdPhysics.MaterialAPI(mat_prim)
            print(f"    staticFriction: {pm.GetStaticFrictionAttr().Get()}")
            print(f"    dynamicFriction: {pm.GetDynamicFrictionAttr().Get()}")
            print(f"    restitution: {pm.GetRestitutionAttr().Get()}")
        if mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            px = PhysxSchema.PhysxMaterialAPI(mat_prim)
            print(f"    frictionCombineMode: {px.GetFrictionCombineModeAttr().Get()}")
    else:
        print("  Bound material: NONE")

    # Check if PhysicsMaterialAPI is on the cube prim directly (should NOT be)
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        pm = UsdPhysics.MaterialAPI(prim)
        print(f"  !! PhysicsMaterialAPI on Cube—staticFriction: {pm.GetStaticFrictionAttr().Get()}")
        print(f"  !! This SHADOWS the bound material's friction values!")
