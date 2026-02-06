#新增mc return字段
"""Add per-episode reward labels to a split HFLibero dataset.

The script assumes the dataset already follows the LeRobot v3.0 layout produced by
``data/split_hflibero_by_suite.py``. It appends a ``reward`` column to every frame,
marking the final ``n_last`` timesteps of each episode with ``reward_value`` and setting
all preceding frames to ``default_reward`` (zero by default).

Usage:
    python data/annotate_rewards.py \
        --dataset-root /path/to/HFlibero_split/libero_goal \
        --output-root /path/to/output/libero_goal_with_rewards \
        --n-last 5 \
        --reward-value 1.0 \
        --default-reward 0.0 
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate rewards on the tail of each episode.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of a split HFlibero dataset (e.g., .../HFlibero_split/libero_goal).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination directory; if provided the dataset is copied here before annotation.",
    )
    parser.add_argument(
        "--n-last",
        type=int,
        default=5,
        help="Number of final timesteps per episode to label with the reward value (default: 1).",
    )
    parser.add_argument(
        "--reward-value",
        type=float,
        default=1.0,
        help="Reward value assigned to the final timesteps (default: 1.0).",
    )
    parser.add_argument(
        "--default-reward",
        type=float,
        default=-1.0,
        help="Reward assigned to non-tail timesteps (default: 0.0).",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.98,
        help="Discount factor used to compute per-frame MC returns (default: 0.98).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing reward column if present. When --output-root is supplied, "
            "also allows replacing an existing destination directory."
        ),
    )
    return parser.parse_args()


def load_episode_bounds(dataset_root: Path) -> Dict[int, Tuple[int, int]]:
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episode metadata directory not found: {episodes_dir}")

    frames = []
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("file-*.parquet")):
            frames.append(pq.read_table(parquet_path).to_pandas())
    if not frames:
        raise RuntimeError(f"No episode metadata parquet files found under {episodes_dir}")

    episodes_df = pd.concat(frames, ignore_index=True)
    bounds: Dict[int, Tuple[int, int]] = {}
    for row in episodes_df.itertuples():
        ep = int(row.episode_index)
        dataset_from = int(row.dataset_from_index)
        dataset_to = int(row.dataset_to_index)
        bounds[ep] = (dataset_from, dataset_to)
    return bounds


def compute_reward_stats(
    bounds: Dict[int, Tuple[int, int]],
    n_last: int,
    reward_value: float,
    default_reward: float,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for ep, (start, end) in bounds.items():
        length = end - start
        if length <= 0:
            stats[ep] = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "count": 0,
            }
            continue

        positive = min(n_last, length)
        negative = length - positive

        if length == 0:
            stats[ep] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
            continue

        mean = (
            positive * reward_value + negative * default_reward
        ) / length
        diff_pos = reward_value - mean
        diff_other = default_reward - mean
        variance = (
            positive * diff_pos * diff_pos + negative * diff_other * diff_other
        ) / length
        std = math.sqrt(variance)

        values = []
        if positive > 0:
            values.append(float(reward_value))
        if negative > 0:
            values.append(float(default_reward))
        if not values:
            values = [0.0]

        stats[ep] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean),
            "std": float(std),
            "count": int(length),
        }
    return stats


def update_info_json(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    if "reward" not in features:
        features["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": info["features"].get("action", {}).get("fps", info.get("fps", 10.0)),
        }
    if "mc_returns" not in features:
        features["mc_returns"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": info["features"].get("action", {}).get("fps", info.get("fps", 10.0)),
        }
    info_path.write_text(json.dumps(info, indent=2))


def write_episode_metadata(
    dataset_root: Path, reward_stats: Dict[int, Dict[str, float]]
) -> None:
    episodes_dir = dataset_root / "meta" / "episodes"
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("file-*.parquet")):
            df = pq.read_table(parquet_path).to_pandas().copy()

            df["stats/reward/min"] = df["episode_index"].map(
                lambda ep: reward_stats[int(ep)]["min"]
            ).astype(np.float32)
            df["stats/reward/max"] = df["episode_index"].map(
                lambda ep: reward_stats[int(ep)]["max"]
            ).astype(np.float32)
            df["stats/reward/mean"] = df["episode_index"].map(
                lambda ep: reward_stats[int(ep)]["mean"]
            ).astype(np.float32)
            df["stats/reward/std"] = df["episode_index"].map(
                lambda ep: reward_stats[int(ep)]["std"]
            ).astype(np.float32)
            df["stats/reward/count"] = df["episode_index"].map(
                lambda ep: reward_stats[int(ep)]["count"]
            ).astype(np.int64)

            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)


def annotate_data_files(
    dataset_root: Path,
    bounds: Dict[int, Tuple[int, int]],
    n_last: int,
    reward_value: float,
    default_reward: float,
    discount: float,
    overwrite: bool,
) -> None:
    data_dir = dataset_root / "data"
    thresholds = {
        ep: max(start, end - n_last) if n_last > 0 else end
        for ep, (start, end) in bounds.items()
    }

    dfs = []
    row_offset = 0
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("file-*.parquet")):
            table = pq.read_table(parquet_path)
            df = table.to_pandas().copy()

            if "reward" in df.columns and not overwrite:
                raise RuntimeError(
                    f"Reward column already present in {parquet_path}. Use --overwrite to replace it."
                )

            indices = df["index"].to_numpy()
            episodes = df["episode_index"].to_numpy()
            threshold_per_row = df["episode_index"].map(thresholds).to_numpy()
            reward = np.where(
                indices >= threshold_per_row, reward_value, default_reward
            ).astype(np.float32)
            df["reward"] = reward

            # Track global row ids so we can stitch MC returns back to each file
            df["_row_id"] = np.arange(row_offset, row_offset + len(df), dtype=np.int64)
            row_offset += len(df)
            dfs.append((parquet_path, df))

    if not dfs:
        return

    all_df = pd.concat([df for _, df in dfs], ignore_index=True)
    mc_returns = np.zeros(len(all_df), dtype=np.float32)
    # Compute discounted returns per episode (sorted by frame index)
    for _, group in all_df.sort_values(["episode_index", "index"]).groupby("episode_index"):
        rewards = group["reward"].to_numpy()
        ret = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = rewards[idx] + discount * running
            ret[idx] = running
        mc_returns[group["_row_id"].to_numpy()] = ret

    # Write back per file
    for parquet_path, df in dfs:
        df["mc_returns"] = mc_returns[df["_row_id"].to_numpy()]
        df = df.drop(columns=["_row_id"])
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)


def main() -> None:
    args = parse_args()
    source_root = args.dataset_root.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")

    if args.output_root is not None:
        target_root = args.output_root.expanduser().resolve()
        if target_root.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output root '{target_root}' already exists. Remove it or rerun with --overwrite "
                "after deleting the destination."
            )
        if target_root != source_root:
            if target_root.exists():
                shutil.rmtree(target_root)
            print(f"[info] Copying dataset to {target_root} ...")
            shutil.copytree(source_root, target_root)
        dataset_root = target_root
    else:
        dataset_root = source_root

    update_info_json(dataset_root)
    bounds = load_episode_bounds(dataset_root)
    reward_stats = compute_reward_stats(bounds, args.n_last, args.reward_value, args.default_reward)

    annotate_data_files(
        dataset_root=dataset_root,
        bounds=bounds,
        n_last=args.n_last,
        reward_value=args.reward_value,
        default_reward=args.default_reward,
        discount=args.discount,
        overwrite=args.overwrite,
    )
    write_episode_metadata(dataset_root, reward_stats)
    print(
        f"[info] Annotated rewards for {len(bounds)} episodes "
        f"({args.n_last} tail frames @ {args.reward_value}, others @ {args.default_reward})."
    )


if __name__ == "__main__":
    main()
