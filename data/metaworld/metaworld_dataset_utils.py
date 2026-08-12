"""Shared utilities for MetaWorld LeRobot dataset processing.

The helpers in this module intentionally mirror the style used by
`data/libero/split_hflibero_by_suite.py` so the two pipelines are easy to merge later.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def iter_parquet_files(root: Path, relative: str) -> Iterator[Path]:
    base = root / relative
    for chunk_dir in sorted(base.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            yield parquet_file


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    return json.loads(info_path.read_text())


def load_tasks(dataset_root: Path) -> pd.DataFrame:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"No tasks.parquet found under {dataset_root}/meta.")
    return pq.read_table(tasks_path).to_pandas()


def load_episodes(dataset_root: Path) -> pd.DataFrame:
    frames = [pq.read_table(path).to_pandas() for path in iter_parquet_files(dataset_root, "meta/episodes")]
    if not frames:
        raise FileNotFoundError(f"No meta/episodes parquet files found under {dataset_root}.")
    return pd.concat(frames, ignore_index=True)


def _extract_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _extract_scalar(value[0])
    if pd.isna(value):
        return None
    return value


def attach_episode_task_id(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized integer task_id to episodes metadata."""

    out = episodes_df.copy()
    if "task_id" in out.columns:
        source_col = "task_id"
    elif "stats/task_id/min" in out.columns:
        source_col = "stats/task_id/min"
    else:
        raise KeyError("Could not infer task_id from episodes metadata (missing task_id columns).")

    out["_task_id"] = out[source_col].map(_extract_scalar)
    if out["_task_id"].isnull().any():
        missing_rows = out[out["_task_id"].isnull()]["episode_index"].tolist()
        raise ValueError(f"Failed to extract task_id for episodes: {missing_rows[:10]}")
    out["_task_id"] = out["_task_id"].astype(int)
    return out


def _choose_episode_ids(
    task_episode_ids: list[int],
    limit: int | None,
    mode: str,
    rng: np.random.Generator,
) -> list[int]:
    if limit is None or limit <= 0 or limit >= len(task_episode_ids):
        return list(task_episode_ids)
    if mode == "sequential":
        return task_episode_ids[:limit]
    indices = rng.choice(len(task_episode_ids), size=min(limit, len(task_episode_ids)), replace=False)
    return sorted(task_episode_ids[idx] for idx in indices)


def select_episode_ids_by_task_id(
    episodes_df: pd.DataFrame,
    few_shot_per_task: int | None,
    few_shot_mode: str,
    rng: np.random.Generator,
    allowed_task_ids: Sequence[int] | None = None,
) -> tuple[list[int], Dict[int, int]]:
    """Return sampled episode ids keyed by task_id."""

    scoped = episodes_df
    if allowed_task_ids is not None:
        allow = {int(x) for x in allowed_task_ids}
        scoped = episodes_df[episodes_df["_task_id"].isin(allow)]
        if scoped.empty:
            raise RuntimeError("No episodes found for the requested --task-ids selection.")

    selected: list[int] = []
    summary: Dict[int, int] = {}
    for task_id in sorted(scoped["_task_id"].unique()):
        rows = scoped[scoped["_task_id"] == task_id].sort_values("episode_index")
        episode_ids = [int(ep) for ep in rows["episode_index"].tolist()]
        chosen = _choose_episode_ids(episode_ids, few_shot_per_task, few_shot_mode, rng)
        selected.extend(chosen)
        summary[int(task_id)] = len(chosen)

    return sorted(set(selected)), summary


def load_frames_for_episodes(
    dataset_root: Path,
    episode_filter: set[int],
    episode_index_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Load and concatenate frame rows for selected episode ids."""

    filtered_frames: list[pd.DataFrame] = []
    target_indices = pa.array(sorted(episode_filter), type=pa.int64())

    for data_file in iter_parquet_files(dataset_root, "data"):
        table = pq.read_table(data_file)
        mask = pc.is_in(table["episode_index"], target_indices)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue

        df = filtered.to_pandas()
        if episode_index_map is not None:
            df["episode_index"] = df["episode_index"].map(episode_index_map).astype("int64")
        filtered_frames.append(df)

    if not filtered_frames:
        raise RuntimeError("No frames matched the filtered episode indices.")

    combined = pd.concat(filtered_frames, ignore_index=True)
    sort_keys = ["episode_index"]
    if "frame_index" in combined.columns:
        sort_keys.append("frame_index")
    elif "index" in combined.columns:
        sort_keys.append("index")

    combined.sort_values(sort_keys, inplace=True)
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


def _wrap_stat_like(sample, value):
    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return np.asarray(value, dtype=sample.dtype)
        return np.asarray([value], dtype=sample.dtype)
    if isinstance(sample, list):
        if len(sample) == 0:
            return [value]
        dtype = type(sample[0])
        return [dtype(value)]
    if isinstance(sample, tuple):
        return (value,)
    return value


def _assign_stat_values(df: pd.DataFrame, column: str, values: list[float | int]) -> None:
    if column not in df.columns:
        return
    sample = None
    for item in df[column].tolist():
        if item is not None:
            sample = item
            break
    if sample is None:
        df[column] = values
        return
    df[column] = [_wrap_stat_like(sample, val) for val in values]


def _sequence_std(start: int, end: int) -> float:
    """Population std for contiguous integer range [start, end)."""

    length = end - start
    if length <= 1:
        return 0.0
    # Variance of {0, ..., n-1} is (n^2 - 1) / 12.
    variance = (length * length - 1) / 12.0
    return math.sqrt(variance)


def update_episodes_metadata(
    episodes_df: pd.DataFrame,
    episode_index_map: dict[int, int],
    episode_lengths: dict[int, int],
    chunk_assignments: dict[int, tuple[int, int]],
) -> pd.DataFrame:
    subset = episodes_df[episodes_df["episode_index"].isin(episode_index_map.keys())].copy()
    subset["episode_index"] = subset["episode_index"].map(episode_index_map).astype("int64")
    subset.sort_values("episode_index", inplace=True)
    subset.reset_index(drop=True, inplace=True)

    dataset_from: list[int] = []
    dataset_to: list[int] = []
    data_chunk: list[int] = []
    data_file: list[int] = []
    lengths: list[int] = []

    cumulative = 0
    for ep_idx in subset["episode_index"].tolist():
        ep_idx = int(ep_idx)
        length = int(episode_lengths[ep_idx])
        start = cumulative
        end = cumulative + length
        cumulative = end

        dataset_from.append(start)
        dataset_to.append(end)
        lengths.append(length)
        chunk_idx, file_idx = chunk_assignments[ep_idx]
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

    _assign_stat_values(subset, "stats/episode_index/min", [int(ep) for ep in subset["episode_index"].tolist()])
    _assign_stat_values(subset, "stats/episode_index/max", [int(ep) for ep in subset["episode_index"].tolist()])
    _assign_stat_values(subset, "stats/episode_index/mean", [float(ep) for ep in subset["episode_index"].tolist()])
    _assign_stat_values(subset, "stats/episode_index/std", [0.0 for _ in range(len(subset))])
    _assign_stat_values(subset, "stats/episode_index/count", [int(length) for length in lengths])

    idx_min = dataset_from
    idx_max = [end - 1 for end in dataset_to]
    idx_mean = [(lo + hi) / 2.0 for lo, hi in zip(idx_min, idx_max)]
    idx_std = [_sequence_std(lo, hi + 1) for lo, hi in zip(idx_min, idx_max)]
    _assign_stat_values(subset, "stats/index/min", idx_min)
    _assign_stat_values(subset, "stats/index/max", idx_max)
    _assign_stat_values(subset, "stats/index/mean", idx_mean)
    _assign_stat_values(subset, "stats/index/std", idx_std)
    _assign_stat_values(subset, "stats/index/count", [int(length) for length in lengths])

    return subset


def update_info(
    info: dict,
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
) -> dict:
    updated = info.copy()
    updated["total_frames"] = int(total_frames)
    updated["total_episodes"] = int(total_episodes)
    updated["total_tasks"] = int(total_tasks)
    updated["splits"] = {"train": f"0:{total_episodes}"}
    return updated


def copy_tasks(dataset_root: Path, destination_root: Path) -> None:
    src = dataset_root / "meta" / "tasks.parquet"
    dst = destination_root / "meta" / "tasks.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_optional_meta_files(dataset_root: Path, destination_root: Path) -> None:
    src = dataset_root / "meta" / "stats.json"
    if src.exists():
        dst = destination_root / "meta" / "stats.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_optional_root_files(dataset_root: Path, destination_root: Path) -> None:
    for name in ("README.md", ".gitattributes"):
        src = dataset_root / name
        if not src.exists():
            continue
        dst = destination_root / name
        shutil.copy2(src, dst)


def load_group_mapping(mapping_json: Path) -> dict[str, list[int]]:
    payload = json.loads(mapping_json.read_text())

    groups_raw = payload.get("task_groups")
    if groups_raw is None:
        groups_raw = payload.get("DIFFICULTY_TO_TASKS")
    if groups_raw is None or not isinstance(groups_raw, dict):
        raise KeyError("Mapping JSON must define `task_groups` or `DIFFICULTY_TO_TASKS`.")

    name_to_id = payload.get("task_name_to_id")
    if name_to_id is None:
        name_to_id = payload.get("TASK_NAME_TO_ID", {})
    if not isinstance(name_to_id, dict):
        raise TypeError("`task_name_to_id` / `TASK_NAME_TO_ID` must be a dictionary when provided.")

    normalized_name_to_id = {str(name): int(task_id) for name, task_id in name_to_id.items()}

    group_to_ids: dict[str, list[int]] = {}
    for group, task_entries in groups_raw.items():
        if not isinstance(task_entries, list):
            raise TypeError(f"Group '{group}' must map to a list of task entries.")

        resolved_ids: list[int] = []
        for entry in task_entries:
            if isinstance(entry, int):
                resolved_ids.append(int(entry))
                continue

            if isinstance(entry, str):
                stripped = entry.strip()
                if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                    resolved_ids.append(int(stripped))
                    continue
                if stripped in normalized_name_to_id:
                    resolved_ids.append(normalized_name_to_id[stripped])
                    continue
                raise KeyError(f"Task entry '{entry}' in group '{group}' cannot be resolved to a task_id.")

            raise TypeError(f"Unsupported task entry type in group '{group}': {type(entry)!r}")

        group_to_ids[str(group)] = sorted(set(resolved_ids))

    return group_to_ids
