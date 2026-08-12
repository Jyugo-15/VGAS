#!/usr/bin/env python
"""Split a merged MetaWorld LeRobot dataset into difficulty groups.

Unlike LIBERO suites, MetaWorld group membership is defined by task_id. This
script keeps task_id semantics intact and only reindexes episode_index.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metaworld_dataset_utils as utils

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_JSON = REPO_ROOT / "data" / "metaworld" / "metaworld_task_groups.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split MetaWorld dataset by task groups.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source merged MetaWorld dataset.")
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON, help="Group mapping JSON.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for per-group datasets.")
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of groups to export (default: all groups in mapping).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing group directories.")
    return parser.parse_args()


def export_group(
    group_name: str,
    group_task_ids: list[int],
    dataset_root: Path,
    output_root: Path,
    info: dict,
    tasks_df,
    episodes_df,
    force: bool,
) -> None:
    group_root = output_root / group_name
    if group_root.exists():
        if not force:
            raise FileExistsError(f"{group_root} already exists. Pass --force to overwrite.")
        shutil.rmtree(group_root)
    group_root.mkdir(parents=True, exist_ok=True)

    subset = episodes_df[episodes_df["_task_id"].isin(set(group_task_ids))].copy()
    if subset.empty:
        raise RuntimeError(f"No episodes found for group '{group_name}' and task_ids={group_task_ids}")

    selected_episode_ids = sorted(int(ep) for ep in subset["episode_index"].tolist())
    episode_index_map = {old: new for new, old in enumerate(selected_episode_ids)}
    episode_filter = set(selected_episode_ids)

    frames = utils.load_frames_for_episodes(
        dataset_root=dataset_root,
        episode_filter=episode_filter,
        episode_index_map=episode_index_map,
    )
    episode_lengths = frames.groupby("episode_index").size().to_dict()
    chunk_assignments = utils.write_episode_data(
        frames=frames,
        destination=group_root / "data",
        chunk_limit=int(info.get("chunks_size", 1000)),
    )

    updated_episodes = utils.update_episodes_metadata(
        episodes_df=episodes_df,
        episode_index_map=episode_index_map,
        episode_lengths=episode_lengths,
        chunk_assignments=chunk_assignments,
    )
    episodes_path = group_root / "meta" / "episodes" / "chunk-000"
    episodes_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(updated_episodes, preserve_index=False),
        episodes_path / "file-000.parquet",
    )

    utils.copy_tasks(dataset_root, group_root)
    info_payload = utils.update_info(
        info=info,
        total_frames=len(frames),
        total_episodes=len(updated_episodes),
        total_tasks=len(tasks_df),
    )
    info_path = group_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info_payload, indent=2))

    utils.copy_optional_meta_files(dataset_root, group_root)
    utils.copy_optional_root_files(dataset_root, group_root)

    print(
        f"[info] exported group '{group_name}' -> {group_root} "
        f"(task_ids={len(group_task_ids)}, episodes={len(updated_episodes)}, frames={len(frames)})"
    )


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    mapping_json = args.mapping_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not mapping_json.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_json}")

    group_to_task_ids = utils.load_group_mapping(mapping_json)
    if args.groups:
        missing = set(args.groups) - set(group_to_task_ids.keys())
        if missing:
            raise ValueError(f"Requested groups {sorted(missing)} not present in mapping JSON.")
        groups_to_export = list(args.groups)
    else:
        groups_to_export = sorted(group_to_task_ids.keys())

    output_root.mkdir(parents=True, exist_ok=True)
    info = utils.load_info(dataset_root)
    tasks_df = utils.load_tasks(dataset_root)
    episodes_df = utils.attach_episode_task_id(utils.load_episodes(dataset_root))

    for group_name in groups_to_export:
        task_ids = group_to_task_ids[group_name]
        if not task_ids:
            print(f"[warn] skip group '{group_name}' because mapping is empty")
            continue
        export_group(
            group_name=group_name,
            group_task_ids=task_ids,
            dataset_root=dataset_root,
            output_root=output_root,
            info=info,
            tasks_df=tasks_df,
            episodes_df=episodes_df,
            force=args.force,
        )


if __name__ == "__main__":
    main()
