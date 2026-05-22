"""
Standalone annotation script: convert recorded HDF5 into MimicGen-compatible format.

Reads the output of record_demo.py and adds `obs/datagen_info/` to each episode:
  - eef_pose: 4x4 EEF pose matrix per step (from obs/eef_pos + obs/eef_quat)
  - object_pose: dict of 4x4 object pose matrices (from obs/cube_pos)
  - target_eef_pose: 4x4 target EEF pose per step (same as eef_pose — actual pose ≈ target)
  - subtask_term_signals: binary signals per subtask (from obs/subtask_terms/)

Usage:
  ./isaaclab.sh -p my_env/annotate_demos.py \
      --input ./datasets/cube_tray_source.hdf5 \
      --output ./datasets/cube_tray_annotated.hdf5
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(
    description="Annotate recorded demos for MimicGen without a Gym env."
)
parser.add_argument(
    "--input", type=str, default="./datasets/cube_tray_source.hdf5",
    help="Input HDF5 from record_demo.py.",
)
parser.add_argument(
    "--output", type=str, default="./datasets/cube_tray_annotated.hdf5",
    help="Output annotated HDF5.",
)
args_cli = parser.parse_args()

import h5py
import numpy as np
import torch

# Tray fixed position (must match scene_config.TRAY_CENTER and MimicEnv cfg)
_TRAY_POS = torch.tensor([-1.71578, -0.0800, 0.1797], dtype=torch.float32)


def pos_quat_to_matrix(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert position (3,) + quaternion wxyz (4,) to 4x4 homogeneous matrix."""
    qw, qx, qy, qz = quat_wxyz
    # Quaternion to rotation matrix
    R = torch.zeros(3, 3, dtype=torch.float32)
    R[0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def main():
    if not os.path.exists(args_cli.input):
        raise FileNotFoundError(f"Input file not found: {args_cli.input}")

    print(f"[INFO] Loading: {args_cli.input}")
    src = h5py.File(args_cli.input, "r")
    data_group = src["data"]

    output_dir = os.path.dirname(args_cli.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    dst = h5py.File(args_cli.output, "w")
    dst_data = dst.create_group("data")

    # Copy env args
    if "env_args" in data_group.attrs:
        dst_data.attrs["env_args"] = data_group.attrs["env_args"]
    dst_data.attrs["total"] = 0

    episode_count = 0
    for ep_name in sorted(data_group.keys()):
        src_ep = data_group[ep_name]
        n_samples = src_ep.attrs["num_samples"]

        print(f"[INFO] Annotating {ep_name} ({n_samples} steps)...")

        # Load source data as torch tensors
        eef_pos = torch.tensor(np.array(src_ep["obs/eef_pos"]))      # (T, 3)
        eef_quat = torch.tensor(np.array(src_ep["obs/eef_quat"]))    # (T, 4) wxyz
        cube_pos = torch.tensor(np.array(src_ep["obs/cube_pos"]))    # (T, 3)

        # Check for subtask terms
        has_subtasks = "obs/subtask_terms" in src_ep
        if has_subtasks:
            st_group = src_ep["obs/subtask_terms"]
            grasp_signal = torch.tensor(np.array(st_group["grasp"])) if "grasp" in st_group else None
            place_signal = torch.tensor(np.array(st_group["place"])) if "place" in st_group else None
        else:
            grasp_signal = None
            place_signal = None

        # Fix grasp signal: the original recording triggers grasp when the hand is
        # within 12cm 3D distance, which can fire too early (hand still 8cm below cube).
        # We shift the grasp trigger to the step where the hand is actually closest
        # to the cube, ensuring the grasp subtask includes the full ascending motion.
        if grasp_signal is not None:
            grasp_on = torch.where(grasp_signal > 0)[0]
            if len(grasp_on) > 0:
                orig_trigger = grasp_on[0].item()
                # Scan forward up to 80 steps to find min z-distance
                search_end = min(orig_trigger + 80, n_samples)
                dz_abs = (eef_pos[orig_trigger:search_end, 2] - cube_pos[orig_trigger:search_end, 2]).abs()
                best_offset = dz_abs.argmin().item()
                corrected_trigger = orig_trigger + best_offset
                if corrected_trigger != orig_trigger:
                    print(f"  grasp signal shifted: step {orig_trigger} → {corrected_trigger} "
                          f"(dz {eef_pos[orig_trigger,2].item() - cube_pos[orig_trigger,2].item():.3f} → "
                          f"{eef_pos[corrected_trigger,2].item() - cube_pos[corrected_trigger,2].item():.3f})")
                    new_grasp = torch.zeros(n_samples, dtype=torch.bool)
                    new_grasp[corrected_trigger:] = True
                    grasp_signal = new_grasp

        # Validate signals have 0→1 transitions (required by DataGenInfoPool)
        skip = False
        for sig_name, sig in [("grasp", grasp_signal), ("place", place_signal)]:
            if sig is None:
                continue
            si = sig.int()
            if (si[1:] - si[:-1] > 0).sum() == 0:
                print(f"  [SKIP] {ep_name}: '{sig_name}' signal never triggers (always {si[0].item()})")
                skip = True
                break
        if skip:
            continue

        # --- Create annotated episode ---
        dst_ep = dst_data.create_group(ep_name)
        dst_ep.attrs["num_samples"] = n_samples
        if "seed" in src_ep.attrs:
            dst_ep.attrs["seed"] = src_ep.attrs["seed"]
        if "success" in src_ep.attrs:
            dst_ep.attrs["success"] = src_ep.attrs["success"]

        # Copy actions
        dst_ep.create_dataset("actions", data=np.array(src_ep["actions"]), compression="gzip")

        # Copy observations (flattened)
        obs_group = dst_ep.create_group("obs")
        for key in src_ep["obs"]:
            if key == "datagen_info":
                continue  # will be recreated
            item = src_ep["obs"][key]
            if isinstance(item, h5py.Group):
                obs_sub = obs_group.create_group(key)
                for sk in item:
                    obs_sub.create_dataset(sk, data=np.array(item[sk]), compression="gzip")
            else:
                obs_group.create_dataset(key, data=np.array(item), compression="gzip")

        # --- Build datagen_info ---
        datagen_group = dst_ep.create_group("obs/datagen_info")

        # eef_pose: (T, 4, 4) for each eef
        eef_pose_group = datagen_group.create_group("eef_pose")
        # Store per-eef: use "left" matching the recording's left hand
        eef_matrices = torch.zeros(n_samples, 4, 4, dtype=torch.float32)
        for t in range(n_samples):
            eef_matrices[t] = pos_quat_to_matrix(eef_pos[t], eef_quat[t])
        eef_pose_group.create_dataset("left", data=eef_matrices.numpy(), compression="gzip")

        # target_eef_pose: use actual eef_pose as target
        target_group = datagen_group.create_group("target_eef_pose")
        target_group.create_dataset("left", data=eef_matrices.numpy(), compression="gzip")

        # object_pose: (T, 4, 4) for each object
        object_group = datagen_group.create_group("object_pose")
        cube_matrices = torch.zeros(n_samples, 4, 4, dtype=torch.float32)
        for t in range(n_samples):
            cube_matrices[t] = pos_quat_to_matrix(
                cube_pos[t],
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # identity rotation
            )
        object_group.create_dataset("cube", data=cube_matrices.numpy(), compression="gzip")
        # Tray is static at a known fixed position
        tray_matrices = torch.zeros(n_samples, 4, 4, dtype=torch.float32)
        for t in range(n_samples):
            tray_matrices[t] = pos_quat_to_matrix(
                _TRAY_POS,
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            )
        object_group.create_dataset("tray", data=tray_matrices.numpy(), compression="gzip")

        # subtask_term_signals: dict of (T,) bool tensors
        term_group = datagen_group.create_group("subtask_term_signals")
        if grasp_signal is not None:
            term_group.create_dataset("grasp", data=grasp_signal.bool().numpy(), compression="gzip")
        if place_signal is not None:
            term_group.create_dataset("place", data=place_signal.bool().numpy(), compression="gzip")

        # Copy initial_state if present
        if "initial_state" in src_ep:
            dst_ep.create_dataset("initial_state", data=np.array(src_ep["initial_state"]), compression="gzip")

        episode_count += 1
        dst_data.attrs["total"] += n_samples

    dst.close()
    src.close()
    print(f"\n[INFO] Annotated {episode_count} episodes → {args_cli.output}")
    if episode_count == 0:
        print("[ERROR] No valid episodes found! Check that source data has grasp+place completions.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
