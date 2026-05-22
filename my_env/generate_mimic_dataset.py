"""
MimicGen data generation launcher for OpenArmX Cube-to-Tray task.

Workflow:
  1. Record source demos:
     ./isaaclab.sh -p my_env/record_demo.py --output ./datasets/source.hdf5

  2. Annotate (auto or manual):
     ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \\
         --task Isaac-OpenArm-Cube-Tray-Mimic-v0 --auto \\
         --input_file ./datasets/source.hdf5 \\
         --output_file ./datasets/source_annotated.hdf5

  3. Generate:
     ./isaaclab.sh -p my_env/generate_mimic_dataset.py \\
         --input_file ./datasets/source_annotated.hdf5 \\
         --output_file ./datasets/cube_tray_generated.hdf5 \\
         --num_envs 4 --generation_num_trials 100
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(
    description="Generate Mimic dataset for OpenArmX Cube-to-Tray task."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--input_file", type=str, required=True,
                    help="Path to annotated source HDF5 dataset.")
parser.add_argument("--output_file", type=str, default="./datasets/cube_tray_generated.hdf5",
                    help="Output path for generated dataset.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments.")
parser.add_argument("--generation_num_trials", type=int, default=100,
                    help="Number of successful demos to generate.")
parser.add_argument("--pause_subtask", action="store_true",
                    help="Pause after each subtask for debugging.")
args_cli = parser.parse_args()

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

    # Build config directly
    env_cfg = _cfg_mod.OpenArmCubeTrayMimicEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.env_name = "Isaac-OpenArm-Cube-Tray-Mimic-v0"
    env_cfg.observations.policy.concatenate_terms = False

    # Output setup
    output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
    output_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Recorders
    env_cfg.recorders = _cfg_mod.OpenArmXRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    if args_cli.generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = args_cli.generation_num_trials

    # Extract + remove success term
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None

    # Create env
    env = _env_mod.OpenArmCubeTrayMimicEnv(cfg=env_cfg)

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("Environment must derive from ManagerBasedRLMimicEnv")

    # Seeds
    random.seed(env_cfg.datagen_config.seed)
    np.random.seed(env_cfg.datagen_config.seed)
    torch.manual_seed(env_cfg.datagen_config.seed)

    env.reset()

    # Setup async generation
    async_components = setup_async_generation(
        env=env,
        num_envs=num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(
            asyncio.gather(*async_components["tasks"])
        )
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Async tasks cleaned up.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
