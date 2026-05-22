"""One-shot script to generate cube_physics.usd with exact recording-like properties.
Run: ./isaaclab.sh -p my_env/make_cube_usd.py

Structure matches recording's create_dynamic_cube() exactly:
  /Cube (UsdGeomCube, RigidBodyAPI+MassAPI+CollisionAPI, material bound directly)
"""
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

stage_path = "/home/huatec/isaac_lab/my_env/cube_physics.usd"
stage = Usd.Stage.CreateNew(stage_path)

# Cube prim — same as create_dynamic_cube: UsdGeomCube with APIs on the same prim
cube = UsdGeom.Cube.Define(stage, "/Cube")
cube.CreateSizeAttr(0.04)
cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.8, 0.0)])
prim = cube.GetPrim()
xf = UsdGeom.Xformable(prim)
xf.ClearXformOpOrder()
xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 2.0))

# Rigid body on the cube prim
rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
rigid_api.CreateRigidBodyEnabledAttr(True)
rigid_api.CreateKinematicEnabledAttr(False)
UsdPhysics.CollisionAPI.Apply(prim)

mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr(0.05)

# Material: green + high friction + combineMode=max
mat = UsdShade.Material.Define(stage, "/Cube/Looks/GreenMaterial")
shader = UsdShade.Shader.Define(stage, "/Cube/Looks/GreenMaterial/PreviewSurface")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.8, 0.0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

phys_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
phys_api.CreateStaticFrictionAttr(20.0)
phys_api.CreateDynamicFrictionAttr(20.0)
phys_api.CreateRestitutionAttr(0.0)
px_api = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
px_api.CreateFrictionCombineModeAttr("max")

# Bind material to the cube prim (same as recording's bind_material)
UsdShade.MaterialBindingAPI(prim).Bind(mat)

stage.GetRootLayer().Save()
print(f"Created: {stage_path}")
