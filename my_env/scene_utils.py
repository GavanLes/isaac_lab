from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from scene_config import CAMERA_PATH_CANDIDATES


def add_sublayer(stage, usd_path):
    """Add a USD file as a sublayer of the current stage."""
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    root_layer = stage.GetRootLayer()
    usd_path_str = str(usd_path)

    if usd_path_str not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(usd_path_str)

    print(f"[INFO] Loaded USD: {usd_path}")


def add_debug_lights(stage):
    """Brighten the scene for debugging."""
    dome = UsdLux.DomeLight.Define(stage, "/World/Debug_DomeLight")
    dome.CreateIntensityAttr(3000.0)
    dome.CreateExposureAttr(0.5)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Debug_KeyLight")
    key.CreateIntensityAttr(6000.0)
    key.CreateAngleAttr(0.5)
    key.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.ClearXformOpOrder()
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, -35.0))

    fill = UsdLux.DistantLight.Define(stage, "/World/Debug_FillLight")
    fill.CreateIntensityAttr(2500.0)
    fill.CreateAngleAttr(0.8)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    fill_xform = UsdGeom.Xformable(fill.GetPrim())
    fill_xform.ClearXformOpOrder()
    fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, 120.0))

    print("[INFO] Added debug lights: DomeLight + KeyLight + FillLight")


def ensure_world(stage):
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")


def find_first_valid_prim(stage, candidates):
    """Find the first existing prim path from a list of candidates."""
    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            print(f"[INFO] Found prim: {path}  type={prim.GetTypeName()}")
            return path

    print("[ERROR] None of these prim paths exist:")
    for path in candidates:
        print("  ", path)

    raise RuntimeError("No valid prim found.")


def get_prim_world_position(stage, prim_path, time=Usd.TimeCode.Default()):
    """Return the world position of a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    world_mat = xform.ComputeLocalToWorldTransform(time)
    pos = world_mat.ExtractTranslation()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


_physx_view_cache = {}


def _get_physx_view(prim_path):
    """Get or create a cached PhysX RigidBodyView, recreating on failure."""
    from isaacsim.core.simulation_manager import SimulationManager

    if prim_path in _physx_view_cache:
        try:
            _physx_view_cache[prim_path].get_transforms()
            return _physx_view_cache[prim_path]
        except Exception:
            del _physx_view_cache[prim_path]

    physics_sim_view = SimulationManager.get_physics_sim_view()
    _physx_view_cache[prim_path] = physics_sim_view.create_rigid_body_view(prim_path)
    return _physx_view_cache[prim_path]


def get_physx_body_position(prim_path):
    """Return the world position of a physics-simulated rigid body via PhysX."""
    view = _get_physx_view(prim_path)
    transforms = view.get_transforms()
    pos = transforms[0, :3]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def reset_physx_body(prim_path, new_pos):
    """Reset a rigid body to a new position with zero velocity via PhysX view."""
    import torch

    # 清除缓存让下次查询重建 view（因为 sim.reset() 会重建物理世界）
    _physx_view_cache.pop(prim_path, None)

    view = _get_physx_view(prim_path)
    device = view._device if hasattr(view, '_device') else 'cpu'

    # pose: (px, py, pz, qx, qy, qz, qw) — identity orientation
    new_pose = torch.tensor(
        [[new_pos[0], new_pos[1], new_pos[2], 0.0, 0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    )
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    view.set_transforms(new_pose, indices=indices)

    zero_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    view.set_velocities(zero_vel, indices=indices)


def set_prim_position(stage, prim_path, pos):
    """Set a prim translate op."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    print(f"[INFO] Set position: {prim_path} -> {pos}")


def create_preview_material(
    stage,
    material_path,
    color,
    roughness=0.45,
    static_friction=None,
    dynamic_friction=None,
    restitution=None,
):
    material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if static_friction is not None or dynamic_friction is not None or restitution is not None:
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        if static_friction is not None:
            physics_material.CreateStaticFrictionAttr(float(static_friction))
        if dynamic_friction is not None:
            physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
        if restitution is not None:
            physics_material.CreateRestitutionAttr(float(restitution))

        # 设置 PhysX 摩擦合并模式为 "max"，确保用摩擦较大一方的值
        # 否则机械爪默认低摩擦会把有效摩擦拉低
        from pxr import PhysxSchema
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr("max")

    return material


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def apply_static_collision(prim):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)


def create_dynamic_cube(stage, cube_path, pos, size, height, mass):
    """Create a green dynamic rectangular prism (cube scaled in Z)."""
    ensure_world(stage)

    if stage.GetPrimAtPath(cube_path).IsValid():
        stage.RemovePrim(cube_path)
        print(f"[INFO] Removed old prim: {cube_path}")

    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(float(size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.8, 0.0)])

    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    z_scale = height / size
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, float(z_scale)))

    material = create_preview_material(
        stage,
        "/World/Materials/Green_Material",
        color=(0.0, 0.8, 0.0),
        static_friction=20.0,
        dynamic_friction=20.0,
        restitution=0.0,
    )
    bind_material(prim, material)

    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        print(f"[INFO] Applied CollisionAPI to: {cube_path}")

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"[INFO] Applied RigidBodyAPI to: {cube_path}")

    rigid_api = UsdPhysics.RigidBodyAPI(prim)
    for create_attr, value in (
        (rigid_api.CreateRigidBodyEnabledAttr, True),
        (rigid_api.CreateKinematicEnabledAttr, False),
        (rigid_api.CreateStartsAsleepAttr, False),
    ):
        try:
            attr = create_attr(value)
            attr.Set(value)
        except Exception:
            pass

    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
    else:
        mass_api = UsdPhysics.MassAPI(prim)
    mass_api.CreateMassAttr(float(mass))
#    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))

    print(f"[INFO] Created dynamic cube: {cube_path}")
    print(f"[INFO] cube pos    = {pos}")
    print(f"[INFO] cube size   = {size}")
    print(f"[INFO] cube height = {height}")
    print(f"[INFO] cube mass   = {mass}")
    return cube_path


def create_box_part(stage, path, pos, scale, material):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.82, 0.78)])

    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

    bind_material(prim, material)
    apply_static_collision(prim)
    return prim


def create_cylinder_part(stage, path, pos, radius, height, material):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.82, 0.78)])

    prim = cylinder.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    bind_material(prim, material)
    apply_static_collision(prim)
    return prim


def create_detection_tray(
    stage,
    tray_path,
    center,
    size_x,
    size_y,
    base_thickness,
    wall_thickness,
    wall_height,
    corner_radius,
):
    """Create a shallow open box used as the target tray."""
    ensure_world(stage)

    if stage.GetPrimAtPath(tray_path).IsValid():
        stage.RemovePrim(tray_path)
        print(f"[INFO] Removed old prim: {tray_path}", flush=True)

    tray_xform = UsdGeom.Xform.Define(stage, tray_path)
    tray_prim = tray_xform.GetPrim()
    tray_xformable = UsdGeom.Xformable(tray_prim)
    tray_xformable.ClearXformOpOrder()
    tray_xformable.AddTranslateOp().Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )

    # Kinematic rigid body so Isaac Lab RigidObject can find it
    if not tray_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(tray_prim)
    else:
        rigid_api = UsdPhysics.RigidBodyAPI(tray_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(True)

    material = create_preview_material(
        stage,
        "/World/Materials/Tray_Material",
        color=(0.82, 0.82, 0.78),
        roughness=0.9,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )

    base_z = 0.0
    wall_z = (base_thickness + wall_height) * 0.5
    radius = min(float(corner_radius), size_x * 0.5, size_y * 0.5)
    square_box = radius <= 0.0

    if square_box:
        create_box_part(
            stage,
            tray_path + "/Base",
            [0.0, 0.0, base_z],
            [size_x, size_y, base_thickness],
            material,
        )
        straight_x = size_x
        straight_y = size_y
    else:
        straight_x = max(size_x - radius * 2.0, 0.001)
        straight_y = max(size_y - radius * 2.0, 0.001)

        create_box_part(
            stage,
            tray_path + "/Base_Long",
            [0.0, 0.0, base_z],
            [straight_x, size_y, base_thickness],
            material,
        )
        create_box_part(
            stage,
            tray_path + "/Base_Wide",
            [0.0, 0.0, base_z],
            [size_x, straight_y, base_thickness],
            material,
        )

        for name, sx, sy in (
            ("Base_Corner_PosX_PosY", 1.0, 1.0),
            ("Base_Corner_PosX_NegY", 1.0, -1.0),
            ("Base_Corner_NegX_PosY", -1.0, 1.0),
            ("Base_Corner_NegX_NegY", -1.0, -1.0),
        ):
            create_cylinder_part(
                stage,
                tray_path + "/" + name,
                [sx * straight_x * 0.5, sy * straight_y * 0.5, base_z],
                radius,
                base_thickness,
                material,
            )

    wall_x_length = straight_x if not square_box else size_x + wall_thickness * 2.0
    wall_y_length = straight_y if not square_box else size_y

    create_box_part(
        stage,
        tray_path + "/Wall_Pos_Y",
        [0.0, size_y * 0.5 + wall_thickness * 0.5, wall_z],
        [wall_x_length, wall_thickness, wall_height],
        material,
    )
    create_box_part(
        stage,
        tray_path + "/Wall_Neg_Y",
        [0.0, -size_y * 0.5 - wall_thickness * 0.5, wall_z],
        [wall_x_length, wall_thickness, wall_height],
        material,
    )
    create_box_part(
        stage,
        tray_path + "/Wall_Pos_X",
        [size_x * 0.5 + wall_thickness * 0.5, 0.0, wall_z],
        [wall_thickness, wall_y_length, wall_height],
        material,
    )
    create_box_part(
        stage,
        tray_path + "/Wall_Neg_X",
        [-size_x * 0.5 - wall_thickness * 0.5, 0.0, wall_z],
        [wall_thickness, wall_y_length, wall_height],
        material,
    )

    if not square_box:
        for name, sx, sy in (
            ("Wall_Corner_PosX_PosY", 1.0, 1.0),
            ("Wall_Corner_PosX_NegY", 1.0, -1.0),
            ("Wall_Corner_NegX_PosY", -1.0, 1.0),
            ("Wall_Corner_NegX_NegY", -1.0, -1.0),
        ):
            create_cylinder_part(
                stage,
                tray_path + "/" + name,
                [
                    sx * (size_x * 0.5 - radius * 0.5),
                    sy * (size_y * 0.5 - radius * 0.5),
                    wall_z,
                ],
                radius + wall_thickness * 0.5,
                wall_height,
                material,
            )

    print(f"[INFO] Created detection tray: {tray_path}", flush=True)
    return tray_path


def choose_front_object_position(robot_pos):
    """Place the cube in front of the robot."""
    return [
        robot_pos[0] + 0.35,
        robot_pos[1] + 0.17,
        robot_pos[2] + 0.20,
    ]


def is_cube_in_tray(
    stage,
    cube_path,
    tray_center,
    tray_size_x,
    tray_size_y,
    tray_wall_height,
    ee_path=None,
):
    """Detect whether the cube center is inside the tray volume.

    If ee_path is given, also requires the end-effector to be outside
    the tray XY bounds, preventing false positives while the gripper
    is still hovering over the tray.
    """
    cube_pos = get_physx_body_position(cube_path)
    dx = abs(cube_pos[0] - tray_center[0])
    dy = abs(cube_pos[1] - tray_center[1])

    inside_xy = dx <= tray_size_x * 0.5 and dy <= tray_size_y * 0.5
    above_floor = cube_pos[2] >= tray_center[2]
    below_wall_top = cube_pos[2] <= tray_center[2] + tray_wall_height

    ee_outside = True
    if ee_path is not None:
        ee_pos = get_prim_world_position(stage, ee_path)
        ee_dx = abs(ee_pos[0] - tray_center[0])
        ee_dy = abs(ee_pos[1] - tray_center[1])
        ee_outside = ee_dx > tray_size_x * 0.5 + 0.03 or ee_dy > tray_size_y * 0.5 + 0.03

    success = inside_xy and above_floor and below_wall_top and ee_outside

    print(
        f"[DETECT] pos=({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})  "
        f"dx={dx:.4f}(<={tray_size_x * 0.5:.4f})  dy={dy:.4f}(<={tray_size_y * 0.5:.4f})  "
        f"inside_xy={inside_xy}  above_floor={above_floor}(>={tray_center[2]:.4f})  "
        f"below_wall_top={below_wall_top}(<={(tray_center[2] + tray_wall_height):.4f})  "
        f"ee_outside={ee_outside}  -> {success}",
        flush=True,
    )

    return success


def reset_cube(cube_path, start_pos):
    """Move the cube back to its start pose and clear velocities via PhysX."""
    reset_physx_body(cube_path, start_pos)
    print("[INFO] Cube entered tray. Scene reset requested.")


def set_joint_positions(
    stage, robot_path, joint_positions,
    arm_stiffness=100000.0, arm_damping=2000.0,
    finger_stiffness=50000.0, finger_damping=500.0,
):
    """Set initial joint positions on a robot articulation by applying drive APIs.

    Finger joints use much higher stiffness to maintain grip on objects.
    """
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        print(f"[WARN] Robot prim not found: {robot_path}")
        return

    applied_count = 0
    for joint_prim in robot_prim.GetAllChildren():
        for prim in [joint_prim] + list(Usd.PrimRange(joint_prim)):
            if not prim.IsA(UsdPhysics.Joint):
                continue

            joint_name = prim.GetName()
            if joint_name not in joint_positions:
                continue

            target = float(joint_positions[joint_name])
            is_finger = "finger" in joint_name

            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "linear")
            else:
                continue

            stiffness = finger_stiffness if is_finger else arm_stiffness
            damping = finger_damping if is_finger else arm_damping
            drive_api.CreateTargetPositionAttr(target)
            drive_api.CreateStiffnessAttr(float(stiffness))
            drive_api.CreateDampingAttr(float(damping))

            print(f"[INFO] Set drive: {prim.GetPath()} -> target={target:.6f}"
                  f" stiffness={stiffness} damping={damping}")
            applied_count += 1

    if applied_count == 0:
        print(f"[WARN] No matching joints found under {robot_path} for provided names")
    else:
        print(f"[INFO] Applied drive to {applied_count} joints under {robot_path}")


def check_prims(stage, robot_path, object_path, tray_path):
    """Check important prims."""
    print("\n[INFO] Checking important prims:")

    paths = [
        robot_path,
        object_path,
        tray_path,
        tray_path + "/Base",
        tray_path + "/Wall_Pos_Y",
        tray_path + "/Wall_Neg_Y",
        tray_path + "/Wall_Pos_X",
        tray_path + "/Wall_Neg_X",
        *CAMERA_PATH_CANDIDATES,
        "/World/Debug_DomeLight",
        "/World/Debug_KeyLight",
        "/World/Debug_FillLight",
    ]

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            print(f"  [OK] {path}    type={prim.GetTypeName()}")
        else:
            print(f"  [MISSING] {path}")


def print_stage_prims(stage, max_count=120):
    """Print part of the stage tree for debugging."""
    print("\n[INFO] Current stage prims:")
    for i, prim in enumerate(stage.Traverse()):
        print(" ", prim.GetPath(), prim.GetTypeName())
        if i >= max_count:
            print("  ...")
            break
