"""Inspect openarmx.usd for finger materials, joint drives, and physics properties."""
from pxr import Usd, UsdShade, UsdPhysics, PhysxSchema

stage = Usd.Stage.Open('/home/huatec/isaac_lab/my_env/openarmx.usd')

# Find all materials with physics properties
print('=== Materials with PhysicsMaterialAPI ===')
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        print(f'  Prim: {prim.GetPath()}')
        api = UsdPhysics.MaterialAPI(prim)
        print(f'    staticFriction: {api.GetStaticFrictionAttr().Get()}')
        print(f'    dynamicFriction: {api.GetDynamicFrictionAttr().Get()}')
        print(f'    restitution: {api.GetRestitutionAttr().Get()}')
        if prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            px = PhysxSchema.PhysxMaterialAPI(prim)
            print(f'    frictionCombineMode: {px.GetFrictionCombineModeAttr().Get()}')

# Find finger link prims and their materials
print()
print('=== Finger-related prims ===')
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if 'finger' in path.lower():
        print(f'  {path}  type={prim.GetTypeName()}')
        mb = UsdShade.MaterialBindingAPI(prim)
        mat, purpose = mb.ComputeBoundMaterial()
        if mat:
            print(f'    bound material: {mat.GetPath()} (purpose={purpose})')
            mat_prim = mat.GetPrim()
            if mat_prim.HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(mat_prim)
                print(f'    staticFriction: {api.GetStaticFrictionAttr().Get()}')
                print(f'    dynamicFriction: {api.GetDynamicFrictionAttr().Get()}')
            if mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
                px = PhysxSchema.PhysxMaterialAPI(mat_prim)
                print(f'    frictionCombineMode: {px.GetFrictionCombineModeAttr().Get()}')

# Find all joints with drives and their stiffness/damping
print()
print('=== Joints with DriveAPI ===')
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joint_name = str(prim.GetPath())
        for drive_type in ['angular', 'linear']:
            try:
                drive = UsdPhysics.DriveAPI(prim, drive_type)
                stiffness = drive.GetStiffnessAttr().Get()
                damping = drive.GetDampingAttr().Get()
                target = drive.GetTargetPositionAttr().Get()
                if stiffness is not None and stiffness > 0:
                    print(f'  {joint_name} ({drive_type}): stiffness={stiffness}, damping={damping}, target={target}')
            except:
                pass
