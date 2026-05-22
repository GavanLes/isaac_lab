"""Inspect a recorded MimicGen HDF5 dataset — shapes, ranges, and sample values.

Usage:
  ./isaaclab.sh -p my_env/inspect_hdf5.py --input ./datasets/cube_tray_source.hdf5
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Inspect MimicGen HDF5 dataset.")
parser.add_argument("--input", type=str, default="./datasets/cube_tray_source.hdf5",
                    help="Path to HDF5 file.")
args_cli = parser.parse_args()

from isaaclab.utils.datasets import HDF5DatasetFileHandler

handler = HDF5DatasetFileHandler()
handler.open(args_cli.input)

print(f"File: {args_cli.input}")
print(f"Env name: {handler.get_env_name()}")
print(f"Num episodes: {handler.get_num_episodes()}\n")

for ep_name in handler.get_episode_names():
    episode = handler.load_episode(ep_name, device="cpu")
    data = episode.data
    print(f"=== {ep_name} ===")
    print(f"  Success: {episode.success}")
    print(f"  Seed:    {episode.seed}")

    actions = data.get("actions")
    if actions is not None:
        print(f"  actions:        shape={actions.shape}  "
              f"min={actions.min().item():.4f}  max={actions.max().item():.4f}")
    else:
        print("  actions:        MISSING")

    obs = data.get("obs")
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    print(f"  obs/{k}/{sk}:  shape={sv.shape}  "
                          f"min={sv.min().item():.4f}  max={sv.max().item():.4f}")
            else:
                print(f"  obs/{k}:        shape={v.shape}  "
                      f"min={v.min().item():.4f}  max={v.max().item():.4f}")

    print()

handler.close()
print("Done.")
