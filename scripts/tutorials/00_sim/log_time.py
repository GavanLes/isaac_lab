# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

"""

"""Launch Isaac Sim Simulator first."""

# 先启动 Isaac Sim 应用，后面的仿真和写日志逻辑都在应用启动后执行。

import argparse
import os

from isaaclab.app import AppLauncher

# create argparser
# 创建命令行参数解析器。
parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")
# append AppLauncher cli args
# 添加 Isaac Lab 通用启动参数，比如 --headless、--device 等。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析命令行参数。
args_cli = parser.parse_args()
# launch omniverse app
# 根据参数启动 Omniverse/Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
# 保存应用对象，用来判断仿真是否仍在运行。
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Sim 启动之后，再导入仿真上下文相关类。
from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    # Specify that the logs must be in logs/docker_tutorial
    # 日志先放在当前工作目录下的 logs 文件夹里。
    log_dir_path = os.path.join("logs")
    # 如果 logs 文件夹不存在，就创建它。
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # In the container, the absolute path will be
    # /workspace/isaaclab/logs/docker_tutorial, because
    # all python execution is done through /workspace/isaaclab/isaaclab.sh
    # and the calling process' path will be /workspace/isaaclab
    # 再进入 logs/docker_tutorial，作为本教程的具体日志目录。
    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    # 如果具体日志目录不存在，就创建它。
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # 打印最终日志目录，方便你知道 log.txt 写到哪里了。
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")

    # Initialize the simulation context
    # 创建仿真配置；dt=0.01 表示每一步仿真时间是 0.01 秒。
    sim_cfg = SimulationCfg(dt=0.01)
    # 创建仿真上下文。
    sim = SimulationContext(sim_cfg)
    # Set main camera
    # 设置主相机视角。
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    # 重置并启动仿真。
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Prepare to count sim_time
    # 获取真实使用的物理时间步长。
    sim_dt = sim.get_physics_dt()
    # 用一个变量手动累计仿真时间。
    sim_time = 0.0

    # Open logging file
    # 打开日志文件；"w" 表示每次运行都会覆盖旧内容。
    with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:
        # Simulate physics
        # 窗口运行期间不断写入时间并推进仿真。
        while simulation_app.is_running():
            # 把当前仿真时间写入 log.txt，每行一个时间值。
            log_file.write(f"{sim_time}" + "\n")
            # perform step
            # 推进一帧仿真。
            sim.step()
            # 累加仿真时间。
            sim_time += sim_dt


if __name__ == "__main__":
    # run the main function
    # 直接运行脚本时执行 main()。
    main()
    # close sim app
    # 退出前关闭 Isaac Sim 应用。
    simulation_app.close()
