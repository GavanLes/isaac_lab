# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""

# 这一段必须放在其它 Isaac Sim/Isaac Lab 模块导入之前，用来先启动 Isaac Sim 应用。

import argparse

from isaaclab.app import AppLauncher

# create argparser
# 创建命令行参数解析器，用来接收运行脚本时传进来的参数。
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
# 添加 Isaac Lab 通用的启动参数，比如 --headless、--device 等。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析命令行参数。
args_cli = parser.parse_args()
# launch omniverse app
# 根据参数启动 Omniverse/Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
# 保存底层的 Isaac Sim 应用对象，后面的循环会用它判断窗口是否还在运行。
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Sim 启动之后，才导入仿真相关的类。
from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Initialize the simulation context
    # 创建仿真配置；dt=0.01 表示物理仿真每一步推进 0.01 秒。
    sim_cfg = SimulationCfg(dt=0.01)
    # 创建仿真上下文，可以理解为当前仿真世界的控制器。
    sim = SimulationContext(sim_cfg)
    # Set main camera
    # 设置主摄像机视角：第一个列表是相机位置，第二个列表是相机看向的位置。
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    # 重置并开始仿真，让 stage 和物理世界进入可运行状态。
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    # 只要 Isaac Sim 窗口还开着，就不断推进仿真。
    while simulation_app.is_running():
        # perform step
        # 推进一帧物理仿真。
        sim.step()


if __name__ == "__main__":
    # run the main function
    # Python 直接运行这个文件时，从 main() 开始执行。
    main()
    # close sim app
    # 脚本结束时关闭 Isaac Sim 应用。
    simulation_app.close()
