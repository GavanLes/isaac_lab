# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""

# 先启动 Isaac Sim 应用；没有这一步，后面的 USD 场景和物理接口不能正常使用。

import argparse

from isaaclab.app import AppLauncher

# create argparser
# 创建命令行参数解析器。
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
# 添加 Isaac Lab 通用启动参数，比如 --headless、--device 等。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析命令行参数。
args_cli = parser.parse_args()
# launch omniverse app
# 根据参数启动 Omniverse/Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
# 保存应用对象，用于判断仿真窗口是否还在运行。
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Sim 启动之后，再导入仿真工具和资源路径。
import isaaclab.sim as sim_utils
# ISAAC_NUCLEUS_DIR 指向 Isaac Sim 自带资源库的位置，里面有很多现成 USD 模型。
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    # 创建地面配置。
    cfg_ground = sim_utils.GroundPlaneCfg()
    # 在 /World/defaultGroundPlane 这个 USD 路径下生成地面。
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    # 创建远光灯配置，用来照亮整个场景。
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    # 在 /World/lightDistant 路径下生成远光灯。
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    # 创建一个 Xform 父节点，后面的物体都挂在 /World/Objects 下面，方便管理。
    sim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone
    # 创建红色圆锥体配置；这里主要是可视化物体，没有刚体和碰撞属性。
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    # 用同一个配置，在两个不同位置生成两个红色圆锥体。
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    # 创建绿色圆锥体配置，并给它添加刚体、质量和碰撞属性。
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        # 刚体属性：让它参与物理运动。
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        # 质量属性：这里设置质量为 1kg。
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        # 碰撞属性：让它能和地面、其它物体发生碰撞。
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    # 生成绿色刚体圆锥体，并设置初始位置和朝向。
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # spawn a blue cuboid with deformable body
    # 创建蓝色可变形长方体配置，它不是普通刚体，而是 deformable body。
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        # 可变形体属性：让物体可以发生形变。
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        # 可变形体使用的物理材质。
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    # 在场景中生成蓝色可变形长方体。
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # spawn a usd file of a table into the scene
    # 从 Isaac 自带资源库里加载一个桌子的 USD 文件。
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # 把桌子生成到 /World/Objects/Table 路径下。
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """Main function."""

    # Initialize the simulation context
    # 创建仿真配置；device 来自命令行参数，比如 cpu/cuda。
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    # 创建仿真上下文。
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    # 设置相机视角。
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    # Design scene
    # 创建场景内容：地面、灯光、圆锥、可变形体、桌子等。
    design_scene()
    # Play the simulator
    # 重置并启动仿真。
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    # 窗口运行期间持续推进仿真。
    while simulation_app.is_running():
        # perform step
        # 推进一帧物理仿真。
        sim.step()


if __name__ == "__main__":
    # run the main function
    # 直接运行脚本时执行 main()。
    main()
    # close sim app
    # 退出前关闭 Isaac Sim 应用。
    simulation_app.close()
