#!/usr/bin/env python3
"""Summarize success rates from MetaWorld eval_info.json artifacts.

Supported directory layouts under ``run_dir``:
    1) <group>/seed_<N>/step_<K>/eval_info.json
    2) <group>/seed_<N>/eval_info.json
    3) <group>/seed_<N>/task_<T>/eval_info.json (fallback)

Example:
    python3 run_scripts/metaworld/summarize_success_rate.py output_metaworld/BC_Only_50_SHOT
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SEED_RE = re.compile(r"^seed_(\d+)$")
STEP_RE = re.compile(r"^step_(\d+)$")
TASK_RE = re.compile(r"^task_(\d+)$")
RUN_STEP_RE = re.compile(r"(?i)(?:^|[_-])step[_-]?(\d+)(?:$|[_-])")
PREFERRED_GROUP_ORDER = ("easy", "medium", "hard", "very_hard")
SUMMARY_META_FIELDS = {"average_by_group", "average_by_task", "n_groups", "n_tasks"}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate success_rate metrics from eval_info.json files.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=Path("output_metaworld/VGAS_5_SHOT_STEP5000_BON8_N_ACTION_20_test"),
        help="Experiment directory, e.g. output_metaworld/BC_Only_50_SHOT",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="success_rate_summary.json",
        help="Output filename saved under run_dir (default: success_rate_summary.json).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Also write one-line-per-(step,seed) text summary.",
    )
    parser.add_argument(
        "--flat-output-name",
        type=str,
        default="success_rate_summary.txt",
        help="Flat output filename saved under run_dir (used with --flat).",
    )
    return parser.parse_args()


def _round6(value):
    if value is None:
        return None
    return round(float(value), 6)


def _task_success_rate(task_entry):
    metrics = task_entry.get("metrics", {})
    successes = metrics.get("successes")
    if isinstance(successes, list) and len(successes) > 0:
        success_count = sum(1 for item in successes if bool(item))
        return 100.0 * success_count / len(successes)
    if "pc_success" in metrics:
        try:
            return float(metrics["pc_success"])
        except (TypeError, ValueError):
            return None
    return None


def _group_success_rate(info, group_name):
    per_group = info.get("per_group", {})
    group_metrics = per_group.get(group_name)

    # Fallback: some files may only contain one group key.
    if group_metrics is None and len(per_group) == 1:
        group_metrics = next(iter(per_group.values()))

    if isinstance(group_metrics, dict) and "pc_success" in group_metrics:
        try:
            return float(group_metrics["pc_success"])
        except (TypeError, ValueError):
            return None

    overall = info.get("overall", {})
    if "pc_success" in overall:
        try:
            return float(overall["pc_success"])
        except (TypeError, ValueError):
            return None
    return None


def _ordered_groups(groups):
    ordered = []
    for group_name in PREFERRED_GROUP_ORDER:
        if group_name in groups:
            ordered.append((group_name, groups[group_name]))
    for group_name in sorted(groups):
        if group_name not in PREFERRED_GROUP_ORDER:
            ordered.append((group_name, groups[group_name]))
    return ordered


def _infer_default_step(run_dir):
    match = RUN_STEP_RE.search(run_dir.name)
    if match is None:
        return 0
    return int(match.group(1))


def summarize_run(run_dir):
    mode = None
    default_step = None
    eval_paths = sorted(run_dir.glob("*/seed_*/step_*/eval_info.json"))
    if eval_paths:
        mode = "step"
    else:
        eval_paths = sorted(run_dir.glob("*/seed_*/eval_info.json"))
        if eval_paths:
            mode = "seed"
            default_step = _infer_default_step(run_dir)
        else:
            eval_paths = sorted(run_dir.glob("*/seed_*/task_*/eval_info.json"))
            if eval_paths:
                mode = "task"
                default_step = _infer_default_step(run_dir)
            else:
                raise FileNotFoundError(
                    f"No eval_info.json found under: {run_dir}\n"
                    "Expected one of:\n"
                    "  - <group>/seed_<N>/step_<K>/eval_info.json\n"
                    "  - <group>/seed_<N>/eval_info.json\n"
                    "  - <group>/seed_<N>/task_<T>/eval_info.json"
                )

    aggregated = defaultdict(
        lambda: {"group_rates": defaultdict(list), "task_rates": []}
    )

    for eval_path in eval_paths:
        rel = eval_path.relative_to(run_dir)
        if mode == "step":
            if len(rel.parts) != 4:
                print(f"[WARN] Skip unexpected path: {eval_path}", file=sys.stderr)
                continue
            group_dir, seed_dir, step_dir, _ = rel.parts
            seed_match = SEED_RE.match(seed_dir)
            step_match = STEP_RE.match(step_dir)
            if seed_match is None or step_match is None:
                print(f"[WARN] Skip invalid seed/step path: {eval_path}", file=sys.stderr)
                continue
            seed = int(seed_match.group(1))
            step = int(step_match.group(1))
        elif mode == "seed":
            if len(rel.parts) != 3:
                print(f"[WARN] Skip unexpected path: {eval_path}", file=sys.stderr)
                continue
            group_dir, seed_dir, _ = rel.parts
            seed_match = SEED_RE.match(seed_dir)
            if seed_match is None:
                print(f"[WARN] Skip invalid seed path: {eval_path}", file=sys.stderr)
                continue
            seed = int(seed_match.group(1))
            step = default_step
        else:
            if len(rel.parts) != 4:
                print(f"[WARN] Skip unexpected path: {eval_path}", file=sys.stderr)
                continue
            group_dir, seed_dir, task_dir, _ = rel.parts
            seed_match = SEED_RE.match(seed_dir)
            task_match = TASK_RE.match(task_dir)
            if seed_match is None or task_match is None:
                print(f"[WARN] Skip invalid seed/task path: {eval_path}", file=sys.stderr)
                continue
            seed = int(seed_match.group(1))
            step = default_step

        try:
            info = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skip invalid JSON {eval_path}: {exc}", file=sys.stderr)
            continue

        key = (step, seed)
        group_rate = _group_success_rate(info, group_dir)
        if group_rate is not None:
            aggregated[key]["group_rates"][group_dir].append(group_rate)

        for task_entry in info.get("per_task", []):
            task_rate = _task_success_rate(task_entry)
            if task_rate is not None:
                aggregated[key]["task_rates"].append(task_rate)

    summary = {}
    for step, seed in sorted(aggregated):
        step_key = f"step_{step:06d}"
        seed_key = f"seed_{seed}"
        group_rates = aggregated[(step, seed)]["group_rates"]
        groups = {
            group_name: (sum(values) / len(values))
            for group_name, values in group_rates.items()
            if values
        }
        task_rates = aggregated[(step, seed)]["task_rates"]

        per_seed = {}
        for group_name, rate in _ordered_groups(groups):
            per_seed[group_name] = _round6(rate)

        average_by_group = (sum(groups.values()) / len(groups)) if groups else None
        average_by_task = (sum(task_rates) / len(task_rates)) if task_rates else None
        per_seed["average_by_group"] = _round6(average_by_group)
        per_seed["average_by_task"] = _round6(average_by_task)
        per_seed["n_groups"] = len(groups)
        per_seed["n_tasks"] = len(task_rates)

        summary.setdefault(step_key, {})[seed_key] = per_seed

    return summary, mode


def _sort_key_from_tag(tag):
    parts = tag.split("_", 1)
    if len(parts) != 2:
        return tag
    suffix = parts[1]
    if suffix.isdigit():
        return int(suffix)
    return suffix


def _fmt_rate(value):
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def summary_to_flat_lines(summary):
    lines = []
    all_group_names = {}
    for seeds in summary.values():
        for per_seed in seeds.values():
            for key in per_seed:
                if key not in SUMMARY_META_FIELDS:
                    all_group_names[key] = None
    ordered_group_names = [name for name, _ in _ordered_groups(all_group_names)]

    for step_key, seeds in sorted(summary.items(), key=lambda item: _sort_key_from_tag(item[0])):
        step_value = step_key.split("_", 1)[1] if "_" in step_key else step_key
        for seed_key, per_seed in sorted(seeds.items(), key=lambda item: _sort_key_from_tag(item[0])):
            seed_value = seed_key.split("_", 1)[1] if "_" in seed_key else seed_key
            group_rates = {
                key: value
                for key, value in per_seed.items()
                if key not in SUMMARY_META_FIELDS
            }
            group_chunks = [f"{group_name}:{_fmt_rate(group_rates.get(group_name))}" for group_name in ordered_group_names]
            strict_avg = None
            if ordered_group_names:
                ordered_values = [group_rates.get(group_name) for group_name in ordered_group_names]
                if all(value is not None for value in ordered_values):
                    strict_avg = sum(float(value) for value in ordered_values) / len(ordered_values)
            avg_chunk = f"avg:{_fmt_rate(strict_avg)}"
            lines.append(f"step:{step_value} seed:{seed_value} {' '.join(group_chunks)} {avg_chunk}".strip())
    return lines


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"[ERROR] Invalid run_dir: {run_dir}", file=sys.stderr)
        return 1

    try:
        summary, mode = summarize_run(run_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    output_path = run_dir / args.output_name
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=args.indent) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Summary written: {output_path}")
    if args.flat:
        flat_lines = summary_to_flat_lines(summary)
        flat_output_path = run_dir / args.flat_output_name
        text = "\n".join(flat_lines)
        if text:
            text += "\n"
        flat_output_path.write_text(text, encoding="utf-8")
        print(f"[OK] Flat summary written: {flat_output_path}")
    print(f"[OK] Layout mode: {mode}")
    print(f"[OK] Steps summarized: {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
