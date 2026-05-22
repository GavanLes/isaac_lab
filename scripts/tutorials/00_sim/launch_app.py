# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run IsaacSim via the AppLauncher

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py

"""

"""Launch Isaac Sim Simulator first."""

# 这一段负责先启动 Isaac Sim 应用，后面的仿真代码都依赖它。

import argparse

from isaaclab.app import AppLauncher

# create argparser
# 创建命令行参数解析器。
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
# 自定义参数：运行脚本时可以用 --size 改变立方体边长。
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.simulation_app/docs/index.html?highlight=simulationapp#isaacsim.simulation_app.SimulationApp
# 自定义窗口/图像宽度参数。
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
# 自定义窗口/图像高度参数。
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)

# append AppLauncher cli args
# 添加 Isaac Lab 通用启动参数，比如 --headless、--device 等。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析所有命令行参数。
args_cli = parser.parse_args()
# launch omniverse app
# 根据参数启动 Omniverse/Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
# 保存应用对象，用于后面的主循环。
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Sim 启动之后，再导入仿真工具模块。
import isaaclab.sim as sim_utils


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    # 创建地面配置。
    cfg_ground = sim_utils.GroundPlaneCfg()
    # 在 USD stage 的 /World/defaultGroundPlane 路径下生成地面。
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    # 创建一个远光灯，类似太阳光，会从远处照亮整个场景。
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    # 把远光灯生成到 /World/lightDistant，并设置它的位置。
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn a cuboid
    # 创建一个立方体配置，边长来自命令行参数 args_cli.size。
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        # 给立方体设置一个白色预览材质。
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    # Spawn cuboid, altering translation on the z-axis to scale to its size
    # 生成立方体；z 坐标设为 size/2，让它刚好落在地面上而不是一半插进地里。
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))


def main():
    """Main function."""

    # Initialize the simulation context
    # 创建仿真配置；dt 是每步物理时间，device 来自命令行参数。
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    # 创建仿真上下文，后面通过它控制场景和物理步进。
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    # 设置主相机：第一个点是相机位置，第二个点是相机看向的位置。
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    # 往场景里添加地面、灯光和立方体。
    design_scene()

    # Play the simulator
    # 重置并启动仿真。
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    # 窗口运行期间，不断推进物理仿真。
    while simulation_app.is_running():
        # perform step
        # 推进一帧仿真。
        sim.step()


if __name__ == "__main__":
    # run the main function
    # 直接运行脚本时执行 main()。
    main()
    # close sim app
    # 退出前关闭 Isaac Sim 应用。
    simulation_app.close()
