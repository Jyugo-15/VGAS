"""Split a merged HFLibero LeRobot dataset into per-suit subsets.

The script reads the merged dataset located at ``--dataset-root`` and creates a
separate LeRobot-formatted dataset for each requested LIBERO task suit. Task
membership is derived from the JSON manifest produced by ``libero_test.py``
(``data/libero_tasksuit_info/libero_task_suites.json``).

Each output dataset contains:
    * Frame parquet files holding only the episodes for the selected suit
      (one file per episode).
    * Updated `meta/info.json`, `meta/tasks.parquet`, and
      `meta/episodes/chunk-000/file-000.parquet`.
    * A copy of `meta/stats.json` when present in the source dataset.

These subsets are directly compatible with Hugging Face Hub uploads.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split HFLibero dataset by LIBERO task suit.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero"),
        help="Path to the merged HFLibero dataset.",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path(
            "/home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/data/libero_tasksuit_info/libero_task_suites.json"
        ),
        help="JSON manifest describing task descriptions per suit and description metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where per-suit datasets will be written.",
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of suits to export (default: all suits present in the mapping).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directories if they already exist.",
    )
    return parser.parse_args()


def iter_parquet_files(root: Path, relative: str) -> Iterator[Path]:
    base = root / relative
    for chunk_dir in sorted(base.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            yield parquet_file


def load_mapping(mapping_path: Path) -> tuple[dict[str, list[str]], dict[str, dict]]:
    payload = json.loads(mapping_path.read_text())
    return payload["task_suites"], payload["description_to_task"]


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    return json.loads(info_path.read_text())


def load_episodes(dataset_root: Path) -> pd.DataFrame:
    frames = [pq.read_table(path).to_pandas() for path in iter_parquet_files(dataset_root, "meta/episodes")]
    if not frames:
        raise FileNotFoundError(f"No meta/episodes parquet files found under {dataset_root}.")
    return pd.concat(frames, ignore_index=True)


def load_tasks(dataset_root: Path) -> pd.DataFrame:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"No tasks.parquet found under {dataset_root}/meta.")
    return pq.read_table(tasks_path).to_pandas()


def _extract_task_description(tasks_value: Sequence[str] | str | np.ndarray) -> str | None:
    if isinstance(tasks_value, np.ndarray):
        tasks_value = tasks_value.tolist()
    if isinstance(tasks_value, (list, tuple)):
        return tasks_value[0] if len(tasks_value) > 0 else None
    return tasks_value


def build_task_maps(
    suits: dict[str, list[str]], description_to_global: dict[str, int]
) -> dict[str, dict[int, int]]:
    suit_maps: dict[str, dict[int, int]] = {}
    for suit, descriptions in suits.items():
        original_indices: list[int] = []
        for desc in descriptions:
            if desc not in description_to_global:
                raise KeyError(
                    f"Description '{desc}' missing from dataset task index mapping. "
                    "Ensure the split manifest matches meta/tasks.parquet."
                )
            original_indices.append(int(description_to_global[desc]))
        suit_maps[suit] = {orig: new for new, orig in enumerate(original_indices)}
    return suit_maps


def load_frames_for_suite(
    dataset_root: Path,
    episode_filter: set[int],
    episode_index_map: dict[int, int],
    task_index_map: dict[int, int],
) -> pd.DataFrame:
    filtered_frames = []
    target_indices = pa.array(list(episode_filter), type=pa.int64())

    for data_file in iter_parquet_files(dataset_root, "data"):
        table = pq.read_table(data_file)
        mask = pc.is_in(table["episode_index"], target_indices)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue
        df = filtered.to_pandas()
        df["episode_index"] = df["episode_index"].map(episode_index_map)
        df["task_index"] = df["task_index"].map(task_index_map)
        filtered_frames.append(df)

    if not filtered_frames:
        raise RuntimeError("No frames matched the filtered episode indices.")

    combined = pd.concat(filtered_frames, ignore_index=True)
    combined.sort_values(["episode_index", "frame_index"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["index"] = combined.index.astype("int64")
    return combined


def write_episode_data(frames: pd.DataFrame, destination: Path, chunk_limit: int) -> dict[int, tuple[int, int]]:
    destination.mkdir(parents=True, exist_ok=True)
    assignments: dict[int, tuple[int, int]] = {}

    chunk_idx = 0
    file_idx = 0

    for ep_idx in sorted(frames["episode_index"].unique()):
        episode_df = frames[frames["episode_index"] == ep_idx]
        chunk_dir = destination / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        file_path = chunk_dir / f"file-{file_idx:03d}.parquet"
        pq.write_table(pa.Table.from_pandas(episode_df, preserve_index=False), file_path)
        assignments[int(ep_idx)] = (chunk_idx, file_idx)

        file_idx += 1
        if file_idx >= chunk_limit:
            chunk_idx += 1
            file_idx = 0

    return assignments


def build_tasks_dataframe(descriptions: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"task_index": range(len(descriptions))}, index=descriptions)
    df.index.name = "task"
    return df


def update_episodes_metadata(
    episodes_df: pd.DataFrame,
    episode_index_map: dict[int, int],
    task_index_map: dict[int, int],
    episode_lengths: dict[int, int],
    chunk_assignments: dict[int, tuple[int, int]],
) -> pd.DataFrame:
    subset = episodes_df[episodes_df["episode_index"].isin(episode_index_map.keys())].copy()
    subset["episode_index"] = subset["episode_index"].map(episode_index_map)
    subset["task_index"] = subset["task_index"].map(task_index_map)
    subset.sort_values("episode_index", inplace=True)
    subset.reset_index(drop=True, inplace=True)

    dataset_from = []
    dataset_to = []
    cumulative = 0

    data_chunk = []
    data_file = []
    lengths = []

    for ep_idx in subset["episode_index"]:
        length = int(episode_lengths[int(ep_idx)])
        dataset_from.append(cumulative)
        cumulative += length
        dataset_to.append(cumulative)
        lengths.append(length)
        chunk_idx, file_idx = chunk_assignments[int(ep_idx)]
        data_chunk.append(chunk_idx)
        data_file.append(file_idx)

    subset["dataset_from_index"] = dataset_from
    subset["dataset_to_index"] = dataset_to
    if "length" in subset.columns:
        subset["length"] = lengths
    else:
        subset.insert(len(subset.columns), "length", lengths)
    subset["data/chunk_index"] = data_chunk
    subset["data/file_index"] = data_file
    subset["meta/episodes/chunk_index"] = 0
    subset["meta/episodes/file_index"] = 0
    return subset


def update_info(info: dict, episodes_df: pd.DataFrame, tasks_df: pd.DataFrame, total_frames: int) -> dict:
    updated = info.copy()
    updated["total_frames"] = int(total_frames)
    updated["total_episodes"] = int(len(episodes_df))
    updated["total_tasks"] = int(len(tasks_df))
    updated["splits"] = {"train": f"0:{len(episodes_df)}"}
    return updated


def copy_stats(dataset_root: Path, destination_root: Path) -> None:
    src = dataset_root / "meta" / "stats.json"
    if src.exists():
        dst = destination_root / "meta" / "stats.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_suite(
    suit: str,
    dataset_root: Path,
    output_root: Path,
    info: dict,
    episodes_df: pd.DataFrame,
    descriptions: list[str],
    description_to_global: dict[str, int],
    task_index_map: dict[int, int],
    force: bool,
) -> None:
    suit_root = output_root / suit
    if suit_root.exists():
        if not force:
            raise FileExistsError(f"{suit_root} already exists. Pass --force to overwrite.")
        shutil.rmtree(suit_root)
    suit_root.mkdir(parents=True, exist_ok=True)

    original_task_indices = [int(description_to_global[desc]) for desc in descriptions]
    episode_filter = set(episodes_df.loc[episodes_df["task_index"].isin(original_task_indices), "episode_index"])
    if not episode_filter:
        raise RuntimeError(f"No episodes found for suit '{suit}'.")
    episode_index_map = {int(old): new for new, old in enumerate(sorted(episode_filter))}

    frames = load_frames_for_suite(
        dataset_root=dataset_root,
        episode_filter=episode_filter,
        episode_index_map=episode_index_map,
        task_index_map=task_index_map,
    )

    episode_lengths = frames.groupby("episode_index").size().to_dict()

    chunk_assignments = write_episode_data(
        frames=frames,
        destination=suit_root / "data",
        chunk_limit=info.get("chunks_size", 1000),
    )

    tasks_df = build_tasks_dataframe(descriptions)
    tasks_path = suit_root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(tasks_path)

    updated_episodes = update_episodes_metadata(
        episodes_df=episodes_df,
        episode_index_map=episode_index_map,
        task_index_map=task_index_map,
        episode_lengths=episode_lengths,
        chunk_assignments=chunk_assignments,
    )

    episodes_path = suit_root / "meta" / "episodes" / "chunk-000"
    episodes_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(updated_episodes, preserve_index=False),
        episodes_path / "file-000.parquet",
    )

    info_payload = update_info(info, updated_episodes, tasks_df, total_frames=len(frames))
    info_path = suit_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info_payload, indent=2))

    copy_stats(dataset_root, suit_root)


def main() -> None:
    args = parse_args()
    task_suites, mapping_metadata = load_mapping(args.mapping_json)

    if args.suites:
        missing = set(args.suites) - set(task_suites.keys())
        if missing:
            raise ValueError(f"Requested suits {missing} not found in mapping manifest.")
        suits_to_export = args.suites
    else:
        suits_to_export = sorted(task_suites.keys())

    info = load_info(args.dataset_root)
    episodes_df = load_episodes(args.dataset_root)
    tasks_df = load_tasks(args.dataset_root)

    description_to_global = {str(index): int(row["task_index"]) for index, row in tasks_df.iterrows()}
    if mapping_metadata:
        missing_from_dataset = sorted(
            desc for desc in mapping_metadata if desc not in description_to_global
        )
        if missing_from_dataset:
            raise KeyError(
                "Descriptions from mapping JSON missing in dataset meta/tasks.parquet: "
                + ", ".join(missing_from_dataset)
            )

    episodes_df["_task_description"] = episodes_df["tasks"].apply(_extract_task_description)
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

    task_maps = build_task_maps(task_suites, description_to_global)

    args.output_root.mkdir(parents=True, exist_ok=True)

    for suit in suits_to_export:
        if suit not in task_maps:
            print(f"[warn] skipping {suit}: no task map available")
            continue
        descriptions = task_suites[suit]
        print(f"[info] exporting suit '{suit}' with {len(descriptions)} tasks")
        export_suite(
            suit=suit,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            info=info,
            episodes_df=episodes_df,
            descriptions=descriptions,
            description_to_global=description_to_global,
            task_index_map=task_maps[suit],
            force=args.force,
        )


if __name__ == "__main__":
    main()
