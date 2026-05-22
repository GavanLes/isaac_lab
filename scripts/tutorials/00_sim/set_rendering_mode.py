# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py

"""

"""Launch Isaac Sim Simulator first."""

# 先启动 Isaac Sim 应用，后面的渲染配置和场景加载都依赖这个应用。

import argparse

from isaaclab.app import AppLauncher

# create argparser
# 创建命令行参数解析器。
parser = argparse.ArgumentParser(
    description="Tutorial on viewing a warehouse scene with a given rendering mode preset."
)
# append AppLauncher cli args
# 添加 Isaac Lab 通用启动参数；如果命令行传了 --rendering_mode，会优先使用命令行值。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析命令行参数。
args_cli = parser.parse_args()
# launch omniverse app
# 根据参数启动 Omniverse/Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
# 保存应用对象，用来判断窗口是否还在运行。
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Sim 启动之后，再导入仿真工具和资源路径。
import isaaclab.sim as sim_utils
# ISAAC_NUCLEUS_DIR 指向 Isaac Sim 自带资源库。
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # rendering modes include performance, balanced, and quality
    # note, the rendering_mode specified in the CLI argument (--rendering_mode) takes precedence over
    # this Render Config setting
    # 渲染模式可以是 performance、balanced、quality；这里默认用性能优先。
    rendering_mode = "performance"

    # carb setting dictionary can include any rtx carb setting which will overwrite the native preset setting
    # carb_settings 可以覆盖底层 RTX 渲染设置；这里强制开启反射。
    carb_settings = {"rtx.reflections.enabled": True}

    # Initialize render config
    # 创建渲染配置，把渲染模式和 RTX 细节设置放进去。
    render_cfg = sim_utils.RenderCfg(
        rendering_mode=rendering_mode,
        carb_settings=carb_settings,
    )

    # Initialize the simulation context with render coofig
    # 创建仿真配置，并把渲染配置传给仿真上下文。
    sim_cfg = sim_utils.SimulationCfg(render=render_cfg)
    # 创建仿真上下文。
    sim = sim_utils.SimulationContext(sim_cfg)

    # Pose camera in the hospital lobby area
    # 设置相机位置，让视角对准医院大厅附近。
    sim.set_camera_view([-11, -0.5, 2], [0, 0, 0.5])

    # Load hospital scene
    # 拼出医院场景 USD 文件的路径。
    hospital_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
    # 创建 USD 文件加载配置。
    cfg = sim_utils.UsdFileCfg(usd_path=hospital_usd_path)
    # 把医院场景加载到 /Scene 路径下。
    cfg.func("/Scene", cfg)

    # Play the simulator
    # 重置并启动仿真。
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run simulation and view scene
    # 窗口运行期间不断推进仿真，这样你可以一直查看场景。
    while simulation_app.is_running():
        # 推进一帧仿真。
        sim.step()


if __name__ == "__main__":
    # run the main function
    # 直接运行脚本时执行 main()。
    main()
    # close sim app
    # 退出前关闭 Isaac Sim 应用。
    simulation_app.close()
