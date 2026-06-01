"""Convert MimicGen HDF5 dataset to LeRobot v3.0 format matching real robot dataset.

Joint mapping (18-dim sim -> 16-dim real):
  sim DOF order (interleaved):
    [L1,R1,L2,R2,L3,R3,L4,R4,L5,R5,L6,R6,L7,R7,l_f1,l_f2,r_f1,r_f2]
  real LeRobot order:
    [L1,L2,L3,L4,L5,L6,L7,l_f1, R1,R2,R3,R4,R5,R6,R7,r_f1]
  keep indices: 0,2,4,6,8,10,12,14,  1,3,5,7,9,11,13,16
  drop indices: 15 (l_finger2), 17 (r_finger2)

Usage: python3 convert_to_lerobot.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_HDF5 = "/home/huatec/isaac_lab/datasets/cube_tray_generated.hdf5"
OUTPUT_DIR = "/home/huatec/isaac_lab/datasets/cube_tray_lerobot"
FPS = 30
BATCH_SIZE = 50  # episodes per batch, keep memory low

# 18-dim (sim) -> 16-dim (real) index mapping
_KEEP_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 16]

# Camera name mapping: sim -> real
_CAM_MAP = {
    "left_hand_cam": "cam_left",
    "right_hand_cam": "cam_right",
    "body_cam": "cam_head",
}

_JOINT_NAMES = [
    "openarmx_left_joint1.pos", "openarmx_left_joint2.pos",
    "openarmx_left_joint3.pos", "openarmx_left_joint4.pos",
    "openarmx_left_joint5.pos", "openarmx_left_joint6.pos",
    "openarmx_left_joint7.pos", "openarmx_left_finger_joint1.pos",
    "openarmx_right_joint1.pos", "openarmx_right_joint2.pos",
    "openarmx_right_joint3.pos", "openarmx_right_joint4.pos",
    "openarmx_right_joint5.pos", "openarmx_right_joint6.pos",
    "openarmx_right_joint7.pos", "openarmx_right_finger_joint1.pos",
]


def _natural_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _quantile_stats(data: np.ndarray) -> dict:
    data = data.astype(np.float64)
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "count": [len(data)],
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q10": np.quantile(data, 0.10, axis=0).tolist(),
        "q50": np.quantile(data, 0.50, axis=0).tolist(),
        "q90": np.quantile(data, 0.90, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


class _VideoEncoder:
    """Long-lived ffmpeg process that receives frames incrementally."""

    def __init__(self, output_path: str, fps: int, H: int, W: int):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._frame_count = 0

    def feed(self, frames: np.ndarray):
        """Feed (T, H, W, 3) uint8 frames to ffmpeg."""
        self.proc.stdin.write(frames.tobytes())
        self._frame_count += frames.shape[0]

    def close(self) -> int:
        """Close stdin, wait for ffmpeg, return total frames encoded."""
        self.proc.stdin.close()
        self.proc.wait()
        return self._frame_count


def _read_episode_data(demo) -> tuple:
    """Read state and action for one episode, converting 18-dim -> 16-dim."""
    states = demo["states"]
    obs = demo["obs"]
    T = demo["actions"].shape[0]

    joint_pos_18 = states["articulation"]["robot"]["joint_position"][:]
    state_16 = joint_pos_18[:, _KEEP_INDICES].astype(np.float32)

    if "joint_pos_target" in demo:
        target_18 = demo["joint_pos_target"][:]
        action_16 = target_18[:, _KEEP_INDICES].astype(np.float32)
        # Finger targets are binary (0 or 0.04). Apply EMA smoothing so the
        # model sees continuous transitions (matching real-robot data).
        # 18-dim: l_f1=14, l_f2=15, r_f1=16, r_f2=17
        # 16-dim after _KEEP_INDICES: l_f1=7, r_f1=15
        for finger_col_18, finger_col_16 in [(14, 7), (16, 15)]:
            raw = target_18[:, finger_col_18].astype(np.float32)
            smoothed = raw.copy()
            alpha = 0.3
            for t in range(1, len(raw)):
                smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t - 1]
            action_16[:, finger_col_16] = smoothed
    else:
        action_16 = state_16.copy()

    return T, state_16, action_16, obs


def main():
    f = h5py.File(INPUT_HDF5, "r")
    demos = sorted(f["data"].keys(), key=_natural_sort_key)
    total_episodes = len(demos)
    print(f"Found {total_episodes} demos")

    # Clean and prepare output directories
    out = Path(OUTPUT_DIR)
    if out.exists():
        shutil.rmtree(out)

    data_dir = out / "data" / "chunk-000"
    video_base = out / "videos"
    meta_dir = out / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: collect episode lengths (metadata only, no heavy data) ----
    print("\n--- Pass 1: collecting episode metadata ---")
    episode_lengths = []
    for demo_name in demos:
        T = f["data"][demo_name]["actions"].shape[0]
        episode_lengths.append(T)

    total_frames = sum(episode_lengths)
    print(f"Total frames: {total_frames}, total episodes: {total_episodes}")

    # ---- Determine video dimensions from first episode ----
    demo0 = f["data"][demos[0]]
    obs0 = demo0["obs"]
    for sim_name, real_name in _CAM_MAP.items():
        if sim_name in obs0:
            sample = obs0[sim_name][0]
            img_h, img_w = sample.shape[0], sample.shape[1]
            break
    print(f"Video dimensions: {img_w}x{img_h}")

    # ---- Create video encoders (one per camera, spans all batches) ----
    video_encoders = {}
    for real_name in _CAM_MAP.values():
        video_dir = video_base / f"observation.images.{real_name}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "file-000.mp4"
        video_encoders[real_name] = _VideoEncoder(str(video_path), FPS, img_h, img_w)

    # ---- Define parquet schema ----
    parquet_schema = pa.schema([
        ("action", pa.list_(pa.float32(), 16)),
        ("observation.state", pa.list_(pa.float32(), 16)),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])

    hf_meta = json.dumps({
        "info": {
            "features": {
                "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 16, "_type": "List"},
                "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 16, "_type": "List"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        },
        "fingerprint": "sim_cube_tray_v1",
    })

    data_path = data_dir / "file-000.parquet"
    parquet_writer = pq.ParquetWriter(str(data_path), schema=parquet_schema)

    # ---- Pass 2: process episodes in batches ----
    print(f"\n--- Pass 2: processing {total_episodes} episodes in batches of {BATCH_SIZE} ---")

    global_offset = 0  # cumulative frame index across all episodes
    ep_rows = []

    for batch_start in range(0, total_episodes, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_episodes)
        batch_episodes = demos[batch_start:batch_end]
        print(f"\n  Batch {batch_start // BATCH_SIZE + 1}: episodes {batch_start}-{batch_end - 1}")

        batch_obs_states = []
        batch_actions = []
        batch_episode_indices = []
        batch_frame_indices = []
        batch_timestamps = []

        for local_idx, demo_name in enumerate(batch_episodes):
            ep_idx = batch_start + local_idx
            demo = f["data"][demo_name]
            T = episode_lengths[ep_idx]

            T_read, state_16, action_16, obs = _read_episode_data(demo)

            batch_obs_states.append(state_16)
            batch_actions.append(action_16)
            batch_episode_indices.append(np.full(T, ep_idx, dtype=np.int64))
            batch_frame_indices.append(np.arange(T, dtype=np.int64))
            batch_timestamps.append(np.arange(T, dtype=np.float32) / FPS)

            # Feed camera frames directly to video encoders (streaming, no accumulation)
            for sim_name, real_name in _CAM_MAP.items():
                if sim_name in obs:
                    video_encoders[real_name].feed(obs[sim_name][:])
                else:
                    black = np.zeros((T, img_h, img_w, 3), dtype=np.uint8)
                    video_encoders[real_name].feed(black)

        # Build flat arrays for this batch
        obs_batch = np.concatenate(batch_obs_states, axis=0)
        act_batch = np.concatenate(batch_actions, axis=0)
        ep_idx_batch = np.concatenate(batch_episode_indices)
        frame_idx_batch = np.concatenate(batch_frame_indices)
        ts_batch = np.concatenate(batch_timestamps)

        batch_frames = obs_batch.shape[0]
        index_batch = np.arange(global_offset, global_offset + batch_frames, dtype=np.int64)
        task_batch = np.zeros(batch_frames, dtype=np.int64)

        # Write this batch as a row group to parquet
        table = pa.table({
            "action": pa.array([row.tolist() for row in act_batch], type=pa.list_(pa.float32(), 16)),
            "observation.state": pa.array([row.tolist() for row in obs_batch], type=pa.list_(pa.float32(), 16)),
            "timestamp": pa.array(ts_batch, type=pa.float32()),
            "frame_index": pa.array(frame_idx_batch, type=pa.int64()),
            "episode_index": pa.array(ep_idx_batch, type=pa.int64()),
            "index": pa.array(index_batch, type=pa.int64()),
            "task_index": pa.array(task_batch, type=pa.int64()),
        })
        parquet_writer.write_table(table)

        # Build per-episode metadata rows for this batch
        dataset_from = global_offset
        for local_idx in range(len(batch_episodes)):
            ep_idx = batch_start + local_idx
            T = episode_lengths[ep_idx]
            dataset_to = dataset_from + T

            row = {
                "episode_index": ep_idx,
                "tasks": ["palce the green cube on the box"],
                "length": T,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": dataset_from,
                "dataset_to_index": dataset_to,
                **{f"videos/observation.images.{rn}/chunk_index": 0 for rn in _CAM_MAP.values()},
                **{f"videos/observation.images.{rn}/file_index": 0 for rn in _CAM_MAP.values()},
                **{f"videos/observation.images.{rn}/from_timestamp": float(dataset_from) / FPS
                   for rn in _CAM_MAP.values()},
                **{f"videos/observation.images.{rn}/to_timestamp": float(dataset_to) / FPS
                   for rn in _CAM_MAP.values()},
                **{f"stats/action/{k}": [float(x) for x in v] if isinstance(v, list) else v
                   for k, v in _quantile_stats(act_batch[dataset_from - global_offset:dataset_to - global_offset]).items()},
                **{f"stats/observation.state/{k}": [float(x) for x in v] if isinstance(v, list) else v
                   for k, v in _quantile_stats(obs_batch[dataset_from - global_offset:dataset_to - global_offset]).items()},
                **{f"stats/observation.images.{rn}/{sk}": [[[0.0]]]
                   for rn in _CAM_MAP.values()
                   for sk in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]},
                **{f"stats/observation.images.{rn}/count": [T] for rn in _CAM_MAP.values()},
                **{f"stats/{col}/{sk}": [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v
                   for col, arr in [
                       ("timestamp", np.arange(T, dtype=np.float32).reshape(-1, 1) / FPS),
                       ("frame_index", np.arange(T, dtype=np.float64).reshape(-1, 1)),
                       ("episode_index", np.full((T, 1), ep_idx, dtype=np.float64)),
                       ("index", np.arange(dataset_from, dataset_to, dtype=np.float64).reshape(-1, 1)),
                       ("task_index", np.zeros((T, 1), dtype=np.float64)),
                   ]
                   for sk, v in _quantile_stats(arr).items()},
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            ep_rows.append(row)
            dataset_from = dataset_to

        global_offset += batch_frames
        print(f"    {batch_frames} frames written")

    f.close()

    # ---- Finalize parquet (add HF metadata) ----
    parquet_writer.close()
    # Re-open to add metadata
    data_table = pq.read_table(str(data_path))
    existing_meta = data_table.schema.metadata or {}
    data_table = data_table.replace_schema_metadata(
        {**existing_meta, b"huggingface": hf_meta.encode()}
    )
    pq.write_table(data_table, str(data_path))
    data_file_size_mb = data_path.stat().st_size / 1e6
    print(f"\n  Parquet written: {data_file_size_mb:.1f} MB")

    # ---- Finalize videos ----
    print("\n--- Finalizing videos ---")
    video_info = {}
    video_total_size = 0
    for real_name, encoder in video_encoders.items():
        total_video_frames = encoder.close()
        video_path = video_base / f"observation.images.{real_name}" / "chunk-000" / "file-000.mp4"
        file_size = video_path.stat().st_size
        duration = total_video_frames / FPS
        video_info[real_name] = {
            "video.height": img_h,
            "video.width": img_w,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
        }
        video_total_size += file_size
        print(f"  {real_name}: {total_video_frames} frames, {file_size / 1e6:.1f} MB, {duration:.1f}s")

    # ---- Compute global stats from parquet (memory-efficient) ----
    print("\n--- Computing global stats ---")
    data_table = pq.read_table(str(data_path))
    obs_all = np.stack([np.asarray(row, dtype=np.float32) for row in data_table.column("observation.state")])
    act_all = np.stack([np.asarray(row, dtype=np.float32) for row in data_table.column("action")])
    ts_all = data_table.column("timestamp").to_numpy().reshape(-1, 1)
    fi_all = data_table.column("frame_index").to_numpy().astype(np.float64).reshape(-1, 1)
    ei_all = data_table.column("episode_index").to_numpy().astype(np.float64).reshape(-1, 1)
    ix_all = data_table.column("index").to_numpy().astype(np.float64).reshape(-1, 1)
    ti_all = data_table.column("task_index").to_numpy().astype(np.float64).reshape(-1, 1)

    global_stats = {
        "action": _quantile_stats(act_all),
        "observation.state": _quantile_stats(obs_all),
        "timestamp": _quantile_stats(ts_all),
        "frame_index": _quantile_stats(fi_all),
        "episode_index": _quantile_stats(ei_all),
        "index": _quantile_stats(ix_all),
        "task_index": _quantile_stats(ti_all),
    }
    for real_name in _CAM_MAP.values():
        ch_stats = {}
        for stat_name in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
            ch_stats[stat_name] = [[[0.0]]]
        ch_stats["count"] = [total_frames]
        global_stats[f"observation.images.{real_name}"] = ch_stats

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    # ---- Write episodes parquet ----
    print("Writing episodes parquet...")
    ep_schema = pa.schema([
        ("episode_index", pa.int64()), ("tasks", pa.list_(pa.string())), ("length", pa.int64()),
        ("data/chunk_index", pa.int64()), ("data/file_index", pa.int64()),
        ("dataset_from_index", pa.int64()), ("dataset_to_index", pa.int64()),
    ] + [
        (f"videos/observation.images.{rn}/{f}", pa.int64() if f.endswith("index") else pa.float64())
        for rn in _CAM_MAP.values()
        for f in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
    ] + [
        (f"stats/action/{sn}", pa.list_(pa.float64()) if sn != "count" else pa.list_(pa.int64()))
        for sn in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    ] + [
        (f"stats/observation.state/{sn}", pa.list_(pa.float64()) if sn != "count" else pa.list_(pa.int64()))
        for sn in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    ] + [
        (f"stats/observation.images.{rn}/{sn}", pa.list_(pa.list_(pa.list_(pa.float64()))))
        for rn in _CAM_MAP.values()
        for sn in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]
    ] + [
        (f"stats/observation.images.{rn}/count", pa.list_(pa.int64()))
        for rn in _CAM_MAP.values()
    ] + [
        (f"stats/{col}/{sn}", pa.list_(pa.float64()) if sn != "count" else pa.list_(pa.int64()))
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        for sn in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    ] + [
        ("meta/episodes/chunk_index", pa.int64()), ("meta/episodes/file_index", pa.int64()),
    ])

    ep_arrays = []
    for field in ep_schema:
        col_name = field.name
        values = [row[col_name] for row in ep_rows]
        ep_arrays.append(pa.array(values, type=field.type))

    ep_table = pa.table(dict(zip([f.name for f in ep_schema], ep_arrays)))
    pq.write_table(ep_table, str(episodes_meta_dir / "file-000.parquet"))

    # ---- Write tasks.parquet ----
    tasks_table = pa.table({
        "task_index": pa.array([0], type=pa.int64()),
        "__index_level_0__": pa.array(["palce the green cube on the box"], type=pa.string()),
    })
    pandas_meta = {
        "index_columns": ["__index_level_0__"],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
            {"name": None, "field_name": "__index_level_0__", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ],
        "attributes": {},
        "creator": {"library": "pyarrow", "version": "23.0.1"},
        "pandas_version": "2.3.3",
    }
    tasks_table = tasks_table.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})
    pq.write_table(tasks_table, str(meta_dir / "tasks.parquet"))

    # ---- Write info.json ----
    info = {
        "codebase_version": "v3.0",
        "robot_type": "openarmx_ros2",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": int(data_file_size_mb) + 1,
        "video_files_size_in_mb": int(video_total_size / 1e6) + 1,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "names": _JOINT_NAMES, "shape": [16]},
            "observation.state": {"dtype": "float32", "names": _JOINT_NAMES, "shape": [16]},
            **{
                f"observation.images.{real_name}": {
                    "dtype": "video",
                    "shape": [img_h, img_w, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": vinfo["video.height"],
                        "video.width": vinfo["video.width"],
                        "video.codec": vinfo["video.codec"],
                        "video.pix_fmt": vinfo["video.pix_fmt"],
                        "video.is_depth_map": False,
                        "video.fps": vinfo["video.fps"],
                        "video.channels": vinfo["video.channels"],
                        "has_audio": False,
                    },
                }
                for real_name, vinfo in video_info.items()
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---- episodes.jsonl ----
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in ep_rows:
            f.write(json.dumps({"episode_index": ep["episode_index"], "length": ep["length"]}) + "\n")

    print(f"\nDone! {total_episodes} episodes, {total_frames} frames")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
