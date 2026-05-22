"""Convert MimicGen HDF5 dataset to LeRobot v3.0 format matching real robot dataset.

Joint mapping (18-dim sim -> 16-dim real):
  sim:    [left_j1-7(7), l_finger1, l_finger2, right_j1-7(7), r_finger1, r_finger2]
  real:   [left_j1-7(7), l_finger1,         right_j1-7(7), r_finger1        ]
  keep indices: 0-6,7,  9-15,16
  drop indices: 8 (l_finger2), 17 (r_finger2)

Usage: python3 convert_to_lerobot.py
"""

import io
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_HDF5 = "/home/huatec/isaac_lab/datasets/cube_tray_generated.hdf5"
OUTPUT_DIR = "/home/huatec/isaac_lab/datasets/cube_tray_lerobot"
FPS = 30

# 18-dim (sim) -> 16-dim (real) index mapping
_KEEP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]

# Camera name mapping: sim -> real
_CAM_MAP = {
    "left_hand_cam": "cam_left",
    "right_hand_cam": "cam_right",
    "body_cam": "cam_head",
}

# Joint names for info.json (16-dim real robot order)
_JOINT_NAMES = [
    "openarmx_left_joint1.pos",
    "openarmx_left_joint2.pos",
    "openarmx_left_joint3.pos",
    "openarmx_left_joint4.pos",
    "openarmx_left_joint5.pos",
    "openarmx_left_joint6.pos",
    "openarmx_left_joint7.pos",
    "openarmx_left_finger_joint1.pos",
    "openarmx_right_joint1.pos",
    "openarmx_right_joint2.pos",
    "openarmx_right_joint3.pos",
    "openarmx_right_joint4.pos",
    "openarmx_right_joint5.pos",
    "openarmx_right_joint6.pos",
    "openarmx_right_joint7.pos",
    "openarmx_right_finger_joint1.pos",
]


def _natural_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _quantile_stats(data: np.ndarray) -> dict:
    """Compute full stats dict matching real robot format."""
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


def _encode_video(frames: np.ndarray, output_path: str, fps: int = FPS) -> dict:
    """Encode (N, H, W, 3) uint8 frames as h264 MP4 via ffmpeg pipe.

    Returns dict with video metadata.
    """
    N, H, W, C = frames.shape
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # Write all frames at once to avoid Python overhead
    raw = frames.tobytes()
    proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()

    duration = N / fps
    file_size = Path(output_path).stat().st_size
    return {
        "video.height": H,
        "video.width": W,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": C,
        "has_audio": False,
        "duration_seconds": duration,
        "file_size_bytes": file_size,
    }


def main():
    f = h5py.File(INPUT_HDF5, "r")
    demos = sorted(f["data"].keys(), key=_natural_sort_key)
    print(f"Found {len(demos)} demos")

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

    # Collect all frames
    all_obs_states = []
    all_actions = []
    all_episode_indices = []
    all_frame_indices = []
    all_timestamps = []

    episodes = []
    camera_frames = {real_name: [] for real_name in _CAM_MAP.values()}
    total_frames = 0

    for ep_idx, demo_name in enumerate(demos):
        demo = f["data"][demo_name]
        obs = demo["obs"]
        states = demo["states"]
        T = demo["actions"].shape[0]

        # Joint positions (measured): 18-dim -> 16-dim
        joint_pos_18 = states["articulation"]["robot"]["joint_position"][:]  # (T, 18)
        state_16 = joint_pos_18[:, _KEEP_INDICES].astype(np.float32)

        # IK target joint positions (action[t] = target for next step): 18-dim -> 16-dim
        if "joint_pos_target" in demo:
            target_18 = demo["joint_pos_target"][:]  # (T, 18)
            action_16 = target_18[:, _KEEP_INDICES].astype(np.float32)
        else:
            # Fallback: use joint position as action (backward compat)
            action_16 = state_16.copy()

        all_obs_states.append(state_16)
        all_actions.append(action_16)
        all_episode_indices.append(np.full(T, ep_idx, dtype=np.int64))
        all_frame_indices.append(np.arange(T, dtype=np.int64))
        all_timestamps.append(np.arange(T, dtype=np.float32) / FPS)

        # Accumulate camera frames
        for sim_name, real_name in _CAM_MAP.items():
            if sim_name in obs:
                camera_frames[real_name].append(obs[sim_name][:])  # (T, H, W, 3) uint8
            else:
                # Missing camera — use black frames
                H, W = 200, 200  # default simulation resolution
                camera_frames[real_name].append(np.zeros((T, H, W, 3), dtype=np.uint8))

        episodes.append({
            "episode_index": ep_idx,
            "length": T,
        })
        total_frames += T
        print(f"  {demo_name}: {T} frames")

    f.close()

    # Build flat arrays
    obs_state_all = np.concatenate(all_obs_states, axis=0)  # (N, 16)
    action_all = np.concatenate(all_actions, axis=0)  # (N, 16)
    episode_idx_all = np.concatenate(all_episode_indices)
    frame_idx_all = np.concatenate(all_frame_indices)
    timestamp_all = np.concatenate(all_timestamps)
    index_all = np.arange(total_frames, dtype=np.int64)
    task_idx_all = np.zeros(total_frames, dtype=np.int64)

    # --- Write data parquet (single file) ---
    print(f"\nWriting data parquet ({total_frames} rows)...")
    obs_col = pa.array([row.tolist() for row in obs_state_all], type=pa.list_(pa.float32(), 16))
    act_col = pa.array([row.tolist() for row in action_all], type=pa.list_(pa.float32(), 16))
    data_table = pa.table({
        "action": act_col,
        "observation.state": obs_col,
        "timestamp": pa.array(timestamp_all, type=pa.float32()),
        "frame_index": pa.array(frame_idx_all, type=pa.int64()),
        "episode_index": pa.array(episode_idx_all, type=pa.int64()),
        "index": pa.array(index_all, type=pa.int64()),
        "task_index": pa.array(task_idx_all, type=pa.int64()),
    })
    # Add huggingface schema metadata (LeRobot requires this)
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
    existing_meta = data_table.schema.metadata or {}
    data_table = data_table.replace_schema_metadata(
        {**existing_meta, b"huggingface": hf_meta.encode()}
    )

    data_path = data_dir / "file-000.parquet"
    pq.write_table(data_table, str(data_path))
    data_file_size_mb = data_path.stat().st_size / 1e6
    print(f"  Written: {data_file_size_mb:.1f} MB")

    # --- Encode videos ---
    print("\nEncoding videos...")
    video_info = {}
    video_total_size = 0
    for real_name in _CAM_MAP.values():
        frames = np.concatenate(camera_frames[real_name], axis=0)  # (N, H, W, 3)
        video_dir = video_base / f"observation.images.{real_name}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "file-000.mp4"
        print(f"  Encoding {real_name}: {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}...")
        vinfo = _encode_video(frames, str(video_path))
        video_info[real_name] = vinfo
        video_total_size += vinfo["file_size_bytes"]
        print(f"    Done: {vinfo['file_size_bytes'] / 1e6:.1f} MB, {vinfo['duration_seconds']:.1f}s")

    # --- Compute global stats ---
    print("\nComputing global stats...")
    global_stats = {
        "action": _quantile_stats(action_all),
        "observation.state": _quantile_stats(obs_state_all),
        "timestamp": _quantile_stats(timestamp_all.reshape(-1, 1)),
        "frame_index": _quantile_stats(frame_idx_all.reshape(-1, 1).astype(np.float64)),
        "episode_index": _quantile_stats(episode_idx_all.reshape(-1, 1).astype(np.float64)),
        "index": _quantile_stats(index_all.reshape(-1, 1).astype(np.float64)),
        "task_index": _quantile_stats(task_idx_all.reshape(-1, 1).astype(np.float64)),
    }
    # Image stats — placeholder (computing per-pixel stats is expensive)
    for real_name in _CAM_MAP.values():
        ch_stats = {}
        for stat_name in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
            ch_stats[stat_name] = [[[0.0]]]
        ch_stats["count"] = [total_frames]
        global_stats[f"observation.images.{real_name}"] = ch_stats

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    # --- Compute per-episode stats and build episodes parquet ---
    print("Computing per-episode stats...")
    ep_rows = []
    dataset_from = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        T = ep["length"]
        dataset_to = dataset_from + T

        # Slice per-episode data
        ep_obs = obs_state_all[dataset_from:dataset_to]
        ep_act = action_all[dataset_from:dataset_to]
        ep_ts = timestamp_all[dataset_from:dataset_to].reshape(-1, 1)
        ep_fi = frame_idx_all[dataset_from:dataset_to].reshape(-1, 1).astype(np.float64)
        ep_ei = episode_idx_all[dataset_from:dataset_to].reshape(-1, 1).astype(np.float64)
        ep_idx_arr = index_all[dataset_from:dataset_to].reshape(-1, 1).astype(np.float64)
        ep_ti = task_idx_all[dataset_from:dataset_to].reshape(-1, 1).astype(np.float64)

        row = {
            "episode_index": ep_idx,
            "tasks": ["palce the green cube on the box"],
            "length": T,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": dataset_from,
            "dataset_to_index": dataset_to,
            # Video references
            **{
                f"videos/observation.images.{real_name}/chunk_index": 0
                for real_name in _CAM_MAP.values()
            },
            **{
                f"videos/observation.images.{real_name}/file_index": 0
                for real_name in _CAM_MAP.values()
            },
            **{
                f"videos/observation.images.{real_name}/from_timestamp": float(dataset_from) / FPS
                for real_name in _CAM_MAP.values()
            },
            **{
                f"videos/observation.images.{real_name}/to_timestamp": float(dataset_to) / FPS
                for real_name in _CAM_MAP.values()
            },
            # Per-episode stats
            **{f"stats/action/{k}": [float(x) for x in v] if isinstance(v, list) else v
               for k, v in _quantile_stats(ep_act).items()},
            **{f"stats/observation.state/{k}": [float(x) for x in v] if isinstance(v, list) else v
               for k, v in _quantile_stats(ep_obs).items()},
            **{f"stats/observation.images.{real_name}/{sk}": [[[0.0]]]
               for real_name in _CAM_MAP.values()
               for sk in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]},
            "stats/observation.images.cam_head/count": [T],
            "stats/observation.images.cam_left/count": [T],
            "stats/observation.images.cam_right/count": [T],
            **{f"stats/{col}/{sk}": [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v
               for col, arr in [
                   ("timestamp", ep_ts), ("frame_index", ep_fi),
                   ("episode_index", ep_ei), ("index", ep_idx_arr), ("task_index", ep_ti),
               ]
               for sk, v in _quantile_stats(arr).items()},
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        ep_rows.append(row)
        dataset_from = dataset_to

    # Build episodes table manually
    ep_schema = pa.schema([
        ("episode_index", pa.int64()),
        ("tasks", pa.list_(pa.string())),
        ("length", pa.int64()),
        ("data/chunk_index", pa.int64()),
        ("data/file_index", pa.int64()),
        ("dataset_from_index", pa.int64()),
        ("dataset_to_index", pa.int64()),
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
        ("meta/episodes/chunk_index", pa.int64()),
        ("meta/episodes/file_index", pa.int64()),
    ])

    # Build pyarrow arrays for each column
    ep_arrays = []
    for field in ep_schema:
        col_name = field.name
        values = [row[col_name] for row in ep_rows]
        ep_arrays.append(pa.array(values, type=field.type))

    ep_table = pa.table(dict(zip([f.name for f in ep_schema], ep_arrays)))
    pq.write_table(ep_table, str(episodes_meta_dir / "file-000.parquet"))

    # --- Write tasks.parquet (with pandas metadata for index column) ---
    tasks_table = pa.table({
        "task_index": pa.array([0], type=pa.int64()),
        "__index_level_0__": pa.array(["palce the green cube on the box"], type=pa.string()),
    })
    # pandas metadata required so LeRobot can resolve task_index -> task name
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

    # --- Write info.json ---
    img_h, img_w = camera_frames["cam_head"][0].shape[1:3]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "openarmx_ros2",
        "total_episodes": len(demos),
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": int(data_file_size_mb) + 1,
        "video_files_size_in_mb": int(video_total_size / 1e6) + 1,
        "fps": FPS,
        "splits": {"train": f"0:{len(demos)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": _JOINT_NAMES,
                "shape": [16],
            },
            "observation.state": {
                "dtype": "float32",
                "names": _JOINT_NAMES,
                "shape": [16],
            },
            **{
                f"observation.images.{real_name}": {
                    "dtype": "video",
                    "shape": [img_h, img_w, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": info_data["video.height"],
                        "video.width": info_data["video.width"],
                        "video.codec": info_data["video.codec"],
                        "video.pix_fmt": info_data["video.pix_fmt"],
                        "video.is_depth_map": False,
                        "video.fps": info_data["video.fps"],
                        "video.channels": info_data["video.channels"],
                        "has_audio": False,
                    },
                }
                for real_name, info_data in video_info.items()
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

    # --- episodes.jsonl (kept for compatibility) ---
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    print(f"\nDone! {len(demos)} episodes, {total_frames} frames")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total size: {shutil.disk_usage(str(out))}")


if __name__ == "__main__":
    main()
