#!/usr/bin/env python
"""Create a task_id few-shot clone of a MetaWorld LeRobot dataset.

This script mirrors the behavior of `data/libero/split_hflibero_few_shot.py` but uses
`task_id` as the sampling unit (MetaWorld MT50 has 50 task_ids while task_index
can be 49 due to shared language descriptions).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metaworld_dataset_utils as utils

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "MetaWorld" / "MT50_50_SHOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce a few-shot clone of a MetaWorld LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Source dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination for few-shot dataset.")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of task ids to keep (defaults to all task ids in the dataset).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists.")
    parser.add_argument(
        "--few-shot-per-task",
        type=int,
        default=None,
        help="Maximum episodes per task_id. Omit to keep all episodes.",
    )
    parser.add_argument(
        "--few-shot-mode",
        choices=["sequential", "random"],
        default="random",
        help="Episode selection strategy when --few-shot-per-task is set.",
    )
    parser.add_argument("--few-shot-seed", type=int, default=0, help="Random seed used when mode=random.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Pass --force to overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = utils.load_info(dataset_root)
    tasks_df = utils.load_tasks(dataset_root)
    episodes_df = utils.attach_episode_task_id(utils.load_episodes(dataset_root))

    rng = np.random.default_rng(args.few_shot_seed)
    episode_ids, summary = utils.select_episode_ids_by_task_id(
        episodes_df=episodes_df,
        few_shot_per_task=args.few_shot_per_task,
        few_shot_mode=args.few_shot_mode,
        rng=rng,
        allowed_task_ids=args.task_ids,
    )
    if not episode_ids:
        raise RuntimeError("Few-shot selection produced zero episodes.")

    episode_index_map = {old: new for new, old in enumerate(episode_ids)}
    episode_filter = set(episode_ids)

    print(f"[info] selected {len(episode_ids)} episodes across {len(summary)} task_ids")
    for task_id in sorted(summary.keys()):
        print(f"    task_id={task_id:02d} | episodes={summary[task_id]:4d}")

    frames = utils.load_frames_for_episodes(
        dataset_root=dataset_root,
        episode_filter=episode_filter,
        episode_index_map=episode_index_map,
    )

    episode_lengths = frames.groupby("episode_index").size().to_dict()
    chunk_assignments = utils.write_episode_data(
        frames=frames,
        destination=output_root / "data",
        chunk_limit=int(info.get("chunks_size", 1000)),
    )

    updated_episodes = utils.update_episodes_metadata(
        episodes_df=episodes_df,
        episode_index_map=episode_index_map,
        episode_lengths=episode_lengths,
        chunk_assignments=chunk_assignments,
    )
    episodes_path = output_root / "meta" / "episodes" / "chunk-000"
    episodes_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(updated_episodes, preserve_index=False),
        episodes_path / "file-000.parquet",
    )

    utils.copy_tasks(dataset_root, output_root)
    info_payload = utils.update_info(
        info=info,
        total_frames=len(frames),
        total_episodes=len(updated_episodes),
        total_tasks=len(tasks_df),
    )
    info_path = output_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info_payload, indent=2))

    utils.copy_optional_meta_files(dataset_root, output_root)
    utils.copy_optional_root_files(dataset_root, output_root)

    print(
        f"[info] wrote few-shot dataset to {output_root} "
        f"({len(summary)} task_ids, {len(episode_ids)} episodes, {len(frames)} frames)."
    )


if __name__ == "__main__":
    main()
