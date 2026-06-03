"""
MimicGen 数据生成启动脚本 — OpenArmX 方块抓取-放置任务

完整流程:
  1. 录制 source demo（键盘遥操或被动模式）:
     ./isaaclab.sh -p my_env/record_demo.py --output ./datasets/source.hdf5

  2. 标注 source demo（自动分割 subtask，标记抓取/放置事件）:
     ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
         --task Isaac-OpenArm-Cube-Tray-Mimic-v0 --auto \
         --input_file ./datasets/source.hdf5 \
         --output_file ./datasets/source_annotated.hdf5

  3. 生成数据集（MimicGen 将 source demo 适配到随机化的新场景）:
     ./isaaclab.sh -p my_env/generate_mimic_dataset.py \
         --input_file ./datasets/source_annotated.hdf5 \
         --output_file ./datasets/cube_tray_generated.hdf5 \
         --num_envs 4 --generation_num_trials 100
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 强制无缓冲输出，确保日志实时可见
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ---- 命令行参数 ----
parser = argparse.ArgumentParser(
    description="生成 Mimic 数据集 — OpenArmX 方块到托盘任务"
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--input_file", type=str, required=True,
                    help="标注后的 source HDF5 数据集路径")
parser.add_argument("--output_file", type=str,
                    default="./datasets/cube_tray_generated.hdf5",
                    help="生成数据集的输出路径")
parser.add_argument("--num_envs", type=int, default=1,
                    help="并行环境数量（>1 可加速生成，但需更多显存）")
parser.add_argument("--generation_num_trials", type=int, default=100,
                    help="需要生成的成功 episode 数量")
parser.add_argument("--pause_subtask", action="store_true",
                    help="每个 subtask 结束后暂停（调试用）")
args_cli = parser.parse_args()

# ---- 启动 Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import asyncio
import random

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation

import openarm_cube_tray_mimic_env as _env_mod
import openarm_cube_tray_mimic_env_cfg as _cfg_mod


def main():
    num_envs = args_cli.num_envs

    # ---- 构建环境配置（与数据生成使用完全相同的 env cfg） ----
    env_cfg = _cfg_mod.OpenArmCubeTrayMimicEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.env_name = "Isaac-OpenArm-Cube-Tray-Mimic-v0"
    env_cfg.observations.policy.concatenate_terms = False  # 分开输出图像和状态

    # ---- 输出路径 ----
    output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
    output_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # ---- 配置数据记录器 ----
    # OpenArmXRecorderManagerCfg 包含自定义的 PostStepJointPosTargetRecorder，
    # 会额外记录每一步的 joint_pos_target（IK 算出的关节目标），
    # 这是 LeRobot 数据集中 action 字段的来源
    env_cfg.recorders = _cfg_mod.OpenArmXRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name

    if env_cfg.datagen_config.generation_keep_failed:
        # 失败和成功的 episode 分开保存
        env_cfg.recorders.dataset_export_mode = \
            DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        # 只保存成功的 episode
        env_cfg.recorders.dataset_export_mode = \
            DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # 命令行可覆盖生成数量
    if args_cli.generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = \
            args_cli.generation_num_trials

    # ---- 提取成功条件后移除 termination ----
    # MimicGen 需要 success_term 来判断 episode 是否成功，
    # 但 env 本身不应因成功而提前终止（生成过程需要完整的轨迹）
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None

    # ---- 创建环境 ----
    # 这里创建的 env 就是训练数据的"真实来源"——
    # 机器人的 PD 参数、物理引擎、渲染管线都从这里决定
    env = _env_mod.OpenArmCubeTrayMimicEnv(cfg=env_cfg)

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("环境必须继承自 ManagerBasedRLMimicEnv")

    # ---- 设置随机种子（保证可复现） ----
    random.seed(env_cfg.datagen_config.seed)
    np.random.seed(env_cfg.datagen_config.seed)
    torch.manual_seed(env_cfg.datagen_config.seed)

    env.reset()

    # ---- 设置异步数据生成管线 ----
    # MimicGen 使用异步架构:
    #   - action_queue: 存放 IK 算好的关节目标
    #   - reset_queue: 需要重置的环境 ID
    #   - info_pool: 源 demo 的 datagen_info 池
    #   - event_loop: asyncio 事件循环
    # 生成器（data_generator）异步计算轨迹 → 放入 action_queue →
    # env_loop 取出执行 → 记录数据
    async_components = setup_async_generation(
        env=env,
        num_envs=num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    try:
        # 启动异步数据生成任务（每个 env 一个协程）
        data_gen_tasks = asyncio.ensure_future(
            asyncio.gather(*async_components["tasks"])
        )
        # 主循环：从 action_queue 取动作 → 执行 → 记录 → 通知生成器
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("任务被取消")
    finally:
        # 清理异步任务
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("异步任务已清理")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()