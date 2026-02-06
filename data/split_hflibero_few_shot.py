#!/usr/bin/env python
"""Create a single few-shot version of the merged HFLibero dataset.

Unlike ``split_hflibero_by_suite.py`` this script does not emit one directory per
LIBERO suit. Instead it copies the entire dataset layout while capping the number
of episodes kept for each task (optionally restricted to a subset of suits).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
import json

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import split_hflibero_by_suite as base

DEFAULT_DATASET_ROOT = Path("/home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero")
DEFAULT_MAPPING_JSON = Path(
    "/home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/data/libero_tasksuit_info/libero_task_suites.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce a few-shot clone of the merged HFLibero dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Merged HFlibero dataset root.")
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=DEFAULT_MAPPING_JSON,
        help="JSON manifest describing LIBERO suits (used to optionally filter tasks).",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Destination directory for the few-shot copy.")
    parser.add_argument(
        "--suites",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of suits to keep. Defaults to all suits in the mapping.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite --output-root if it already exists.")
    parser.add_argument(
        "--few-shot-per-task",
        type=int,
        default=None,
        help="Maximum number of episodes per task. When omitted, all episodes are kept.",
    )
    parser.add_argument(
        "--few-shot-mode",
        choices=["sequential", "random"],
        default="sequential",
        help="Episode selection strategy when --few-shot-per-task is set.",
    )
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        default=0,
        help="Random seed used when --few-shot-mode=random.",
    )
    return parser.parse_args()


def _select_episode_ids(
    episodes_df: pd.DataFrame,
    limit: int | None,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[List[int], Dict[int, int]]:
    """Return per-task few-shot selections."""

    selected: List[int] = []
    summary: Dict[int, int] = {}
    for task_idx in sorted(episodes_df["task_index"].unique()):
        task_rows = episodes_df[episodes_df["task_index"] == task_idx].sort_values("episode_index")
        episode_ids = task_rows["episode_index"].tolist()
        if limit is None or limit <= 0 or limit >= len(episode_ids):
            chosen = episode_ids
        elif mode == "sequential":
            chosen = episode_ids[:limit]
        else:
            indices = rng.choice(len(episode_ids), size=min(limit, len(episode_ids)), replace=False)
            chosen = sorted(task_rows.iloc[indices]["episode_index"].tolist())
        summary[int(task_idx)] = len(chosen)
        selected.extend(int(ep) for ep in chosen)
    unique = sorted(set(selected))
    return unique, summary


def _resolve_task_filter(
    task_suites: dict[str, list[str]],
    selected_suits: Sequence[str] | None,
    description_to_global: Dict[str, int],
) -> List[int]:
    if selected_suits is None:
        descriptions: list[str] = []
        for suit in sorted(task_suites.keys()):
            descriptions.extend(task_suites[suit])
    else:
        missing = set(selected_suits) - set(task_suites.keys())
        if missing:
            raise ValueError(f"Requested suits {missing} not present in mapping JSON.")
        descriptions = []
        for suit in selected_suits:
            descriptions.extend(task_suites[suit])
    indices = []
    for desc in descriptions:
        if desc not in description_to_global:
            raise KeyError(f"Task description '{desc}' missing from dataset meta/tasks.parquet.")
        indices.append(int(description_to_global[desc]))
    return sorted(set(indices))


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_root} already exists. Pass --force to overwrite.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    task_suites, mapping_metadata = base.load_mapping(args.mapping_json)
    info = base.load_info(args.dataset_root)
    episodes_df = base.load_episodes(args.dataset_root)
    tasks_df = base.load_tasks(args.dataset_root)

    description_to_global = {str(index): int(row["task_index"]) for index, row in tasks_df.iterrows()}
    if mapping_metadata:
        missing_desc = sorted(desc for desc in mapping_metadata if desc not in description_to_global)
        if missing_desc:
            raise KeyError(
                "Descriptions from mapping JSON missing in dataset meta/tasks.parquet: " + ", ".join(missing_desc)
            )

    episodes_df["_task_description"] = episodes_df["tasks"].apply(base._extract_task_description)
    episodes_df["_task_index_original"] = episodes_df["_task_description"].map(
        lambda desc: int(description_to_global[desc]) if desc in description_to_global else None
    )
    if episodes_df["_task_index_original"].isnull().any():
        missing = episodes_df.loc[episodes_df["_task_index_original"].isnull(), "_task_description"].unique()
        raise KeyError("Missing task descriptions in mapping: " + ", ".join(sorted(map(str, missing))))
    episodes_df["task_index"] = episodes_df["_task_index_original"].astype(int)

    selected_task_indices = _resolve_task_filter(task_suites, args.suites, description_to_global)
    if not selected_task_indices:
        raise RuntimeError("No tasks matched the requested suits.")
    mask = episodes_df["task_index"].isin(selected_task_indices)
    filtered_episodes = episodes_df[mask].copy()
    if filtered_episodes.empty:
        raise RuntimeError("No episodes found for the requested task selection.")

    rng = np.random.default_rng(args.few_shot_seed)
    episode_ids, summary = _select_episode_ids(filtered_episodes, args.few_shot_per_task, args.few_shot_mode, rng)
    if not episode_ids:
        raise RuntimeError("Few-shot filtering removed all episodes.")

    active_tasks = [task for task, count in sorted(summary.items()) if count > 0]
    if not active_tasks:
        raise RuntimeError("Few-shot configuration produced zero tasks.")

    global_to_description = {idx: desc for desc, idx in description_to_global.items()}
    descriptions = [global_to_description[idx] for idx in active_tasks]
    task_index_map = {old: new for new, old in enumerate(active_tasks)}
    episode_index_map = {old: new for new, old in enumerate(episode_ids)}
    episode_filter = set(episode_ids)

    print(f"[info] keeping {len(active_tasks)} tasks with few-shot sampling")
    for task_idx in active_tasks:
        desc = global_to_description.get(task_idx, "<unknown>")
        kept = summary.get(task_idx, 0)
        print(f"    task_index={task_idx:03d} | episodes={kept:4d} | desc={desc}")

    frames = base.load_frames_for_suite(
        dataset_root=args.dataset_root,
        episode_filter=episode_filter,
        episode_index_map=episode_index_map,
        task_index_map=task_index_map,
    )

    episode_lengths = frames.groupby("episode_index").size().to_dict()
    chunk_assignments = base.write_episode_data(
        frames=frames,
        destination=args.output_root / "data",
        chunk_limit=info.get("chunks_size", 1000),
    )

    tasks_payload = base.build_tasks_dataframe(descriptions)
    tasks_path = args.output_root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_payload.to_parquet(tasks_path)

    updated_episodes = base.update_episodes_metadata(
        episodes_df=episodes_df,
        episode_index_map=episode_index_map,
        task_index_map=task_index_map,
        episode_lengths=episode_lengths,
        chunk_assignments=chunk_assignments,
    )
    episodes_path = args.output_root / "meta" / "episodes" / "chunk-000"
    episodes_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(updated_episodes, preserve_index=False), episodes_path / "file-000.parquet")

    info_payload = base.update_info(info, updated_episodes, tasks_payload, total_frames=len(frames))
    info_path = args.output_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info_payload, indent=2))

    base.copy_stats(args.dataset_root, args.output_root)
    print(
        f"[info] wrote few-shot dataset to {args.output_root} "
        f"({len(active_tasks)} tasks, {len(episode_ids)} episodes)."
    )


if __name__ == "__main__":
    main()
