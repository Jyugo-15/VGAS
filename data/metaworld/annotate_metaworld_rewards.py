#!/usr/bin/env python
"""Annotate custom reward labels for MetaWorld LeRobot datasets.

This script intentionally does not compute/write MC returns.
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
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination directory. If set, source dataset is copied first.",
    )
    parser.add_argument("--n-last", type=int, default=3, help="Final timesteps per episode marked as reward-value.")
    parser.add_argument("--reward-value", type=float, default=1.0, help="Reward for final timesteps.")
    parser.add_argument("--default-reward", type=float, default=-1.0, help="Reward for all other timesteps.")
    parser.add_argument(
        "--terminal-mode",
        choices=["none", "episode_end", "next_success"],
        default="none",
        help="How to create terminal flag.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite reward/terminal columns if they already exist.",
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
        mean = (positive * reward_value + negative * default_reward) / length
        diff_pos = reward_value - mean
        diff_other = default_reward - mean
        variance = (positive * diff_pos * diff_pos + negative * diff_other * diff_other) / length
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


def update_info_json(dataset_root: Path, include_terminal: bool) -> None:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})

    if "reward" not in features:
        features["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": info.get("fps", 80),
        }

    if include_terminal and "terminal" not in features:
        features["terminal"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
            "fps": info.get("fps", 80),
        }

    info_path.write_text(json.dumps(info, indent=2))


def write_episode_metadata(dataset_root: Path, reward_stats: Dict[int, Dict[str, float]]) -> None:
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
    terminal_mode: str,
    overwrite: bool,
) -> None:
    data_dir = dataset_root / "data"

    thresholds = {
        ep: max(start, end - n_last) if n_last > 0 else end
        for ep, (start, end) in bounds.items()
    }
    end_indices = {ep: end - 1 for ep, (_, end) in bounds.items()}

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("file-*.parquet")):
            table = pq.read_table(parquet_path)
            df = table.to_pandas().copy()

            if "reward" in df.columns and not overwrite:
                raise RuntimeError(
                    f"Reward column already present in {parquet_path}. Use --overwrite to replace it."
                )

            indices = df["index"].to_numpy()
            threshold_per_row = df["episode_index"].map(thresholds).to_numpy()
            reward = np.where(indices >= threshold_per_row, reward_value, default_reward).astype(np.float32)
            df["reward"] = reward

            if terminal_mode != "none":
                if "terminal" in df.columns and not overwrite:
                    raise RuntimeError(
                        f"Terminal column already present in {parquet_path}. Use --overwrite to replace it."
                    )

                if terminal_mode == "episode_end":
                    end_per_row = df["episode_index"].map(end_indices).to_numpy()
                    terminal = (indices == end_per_row)
                else:
                    if "next.success" not in df.columns:
                        raise KeyError(
                            f"Column 'next.success' not found in {parquet_path}; cannot use --terminal-mode next_success."
                        )
                    terminal = df["next.success"].fillna(False).astype(bool).to_numpy()

                df["terminal"] = terminal.astype(bool)

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
                f"Output root '{target_root}' already exists. Remove it or rerun with --overwrite."
            )
        if target_root != source_root:
            if target_root.exists():
                shutil.rmtree(target_root)
            print(f"[info] Copying dataset to {target_root} ...")
            shutil.copytree(source_root, target_root)
        dataset_root = target_root
    else:
        dataset_root = source_root

    include_terminal = args.terminal_mode != "none"
    update_info_json(dataset_root, include_terminal=include_terminal)

    bounds = load_episode_bounds(dataset_root)
    reward_stats = compute_reward_stats(bounds, args.n_last, args.reward_value, args.default_reward)

    annotate_data_files(
        dataset_root=dataset_root,
        bounds=bounds,
        n_last=args.n_last,
        reward_value=args.reward_value,
        default_reward=args.default_reward,
        terminal_mode=args.terminal_mode,
        overwrite=args.overwrite,
    )
    write_episode_metadata(dataset_root, reward_stats)

    print(
        f"[info] Annotated rewards for {len(bounds)} episodes "
        f"({args.n_last} tail frames @ {args.reward_value}, others @ {args.default_reward}; "
        f"terminal_mode={args.terminal_mode})."
    )


if __name__ == "__main__":
    main()
