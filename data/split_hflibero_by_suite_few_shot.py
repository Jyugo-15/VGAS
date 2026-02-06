#!/usr/bin/env python
"""Few-shot wrapper around split_hflibero_by_suite.

Creates per-suit datasets like the original splitter but limits each task
to a fixed number of episodes either deterministically (first N) or via
random sampling without replacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import split_hflibero_by_suite as base

DEFAULT_DATASET_ROOT = Path("/home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero")
DEFAULT_MAPPING_JSON = Path(
    "/home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/data/libero_tasksuit_info/libero_task_suites.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split HFLibero suits with optional few-shot sampling.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Merged HFLibero dataset root.")
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=DEFAULT_MAPPING_JSON,
        help="JSON manifest describing task descriptions per suit.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for per-suit datasets.")
    parser.add_argument(
        "--suites",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of suits to export (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite suit directories if they already exist.")
    parser.add_argument(
        "--few-shot-per-task",
        type=int,
        default=10,
        help="Limit each task to at most N episodes. When omitted all episodes are kept.",
    )
    parser.add_argument(
        "--few-shot-mode",
        type=str,
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
    suit_df: pd.DataFrame,
    limit: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[List[int], Dict[int, int]]:
    selected: List[int] = []
    summary: Dict[int, int] = {}
    for task_idx in sorted(suit_df["task_index"].unique()):
        task_rows = suit_df[suit_df["task_index"] == task_idx].sort_values("episode_index")
        episode_ids = task_rows["episode_index"].tolist()
        if limit >= len(episode_ids):
            chosen = episode_ids
        elif mode == "sequential":
            chosen = episode_ids[:limit]
        else:  # random
            indices = rng.choice(len(episode_ids), size=limit, replace=False)
            chosen = sorted(task_rows.iloc[indices]["episode_index"].tolist())
        summary[int(task_idx)] = len(chosen)
        selected.extend(chosen)
    return sorted(set(int(ep) for ep in selected)), summary


def _prepare_episodes_for_suit(
    episodes_df: pd.DataFrame,
    original_task_indices: Iterable[int],
    few_shot_limit: int | None,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    mask = episodes_df["task_index"].isin(list(original_task_indices))
    suit_subset = episodes_df[mask].copy()
    if suit_subset.empty:
        raise RuntimeError("No episodes found for requested tasks.")
    if few_shot_limit is None:
        counts = suit_subset.groupby("task_index").size().to_dict()
        return suit_subset, {int(idx): int(count) for idx, count in counts.items()}

    if few_shot_limit <= 0:
        raise ValueError("--few-shot-per-task must be positive when provided.")

    selected_ids, summary = _select_episode_ids(suit_subset, few_shot_limit, mode, rng)
    filtered = suit_subset[suit_subset["episode_index"].isin(selected_ids)].copy()
    if filtered.empty:
        raise RuntimeError("Few-shot filtering removed all episodes for suit.")
    return filtered, summary


def main() -> None:
    args = parse_args()

    task_suites, mapping_metadata = base.load_mapping(args.mapping_json)
    if args.suites:
        missing = set(args.suites) - set(task_suites.keys())
        if missing:
            raise ValueError(f"Requested suits {missing} not found in mapping manifest.")
        suits_to_export = args.suites
    else:
        suits_to_export = sorted(task_suites.keys())

    info = base.load_info(args.dataset_root)
    episodes_df = base.load_episodes(args.dataset_root)
    tasks_df = base.load_tasks(args.dataset_root)

    description_to_global = {str(index): int(row["task_index"]) for index, row in tasks_df.iterrows()}
    if mapping_metadata:
        missing_from_dataset = sorted(desc for desc in mapping_metadata if desc not in description_to_global)
        if missing_from_dataset:
            raise KeyError(
                "Descriptions from mapping JSON missing in dataset meta/tasks.parquet: "
                + ", ".join(missing_from_dataset)
            )

    episodes_df["_task_description"] = episodes_df["tasks"].apply(base._extract_task_description)
    episodes_df["_task_index_original"] = episodes_df["_task_description"].map(
        lambda desc: int(description_to_global[desc]) if desc in description_to_global else None
    )
    if episodes_df["_task_index_original"].isnull().any():
        missing_desc = episodes_df.loc[episodes_df["_task_index_original"].isnull(), "_task_description"].unique()
        raise KeyError(
            "Some episode task descriptions are absent from the dataset task mapping: "
            + ", ".join(sorted(map(str, missing_desc)))
        )
    episodes_df["task_index"] = episodes_df["_task_index_original"].astype(int)

    task_maps = base.build_task_maps(task_suites, description_to_global)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.few_shot_seed)

    for suit in suits_to_export:
        descriptions = task_suites[suit]
        original_task_indices = [int(description_to_global[desc]) for desc in descriptions]
        suit_episodes, summary = _prepare_episodes_for_suit(
            episodes_df,
            original_task_indices,
            args.few_shot_per_task,
            args.few_shot_mode,
            rng,
        )
        kept = suit_episodes["episode_index"].nunique()
        print(f"[info] exporting suit '{suit}' with {len(descriptions)} tasks (keeping {kept} episodes)")
        for task_idx, count in sorted(summary.items()):
            print(f"    task_index={task_idx}: {count} episodes")
        base.export_suite(
            suit=suit,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            info=info,
            episodes_df=suit_episodes,
            descriptions=descriptions,
            description_to_global=description_to_global,
            task_index_map=task_maps[suit],
            force=args.force,
        )


if __name__ == "__main__":
    main()
