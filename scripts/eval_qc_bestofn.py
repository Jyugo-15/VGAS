"""Evaluate SmolVLA with optional Best-of-N critic guidance."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

from huggingface_hub.constants import CONFIG_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
# Keep local lerobot as a fallback path, but do not shadow installed package versions.
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.append(str(LEROBOT_SRC))

DEFAULT_POLICY_PATH = REPO_ROOT / "pretrained_vla/smolvla/5-shot/pretrained_model"
DEFAULT_CRITIC_PATH = REPO_ROOT / "outputs/train/12.27/goal_vgas/checkpoints/018000/critic_pretrained_model/last.ckpt"

try:
    from libero.libero import benchmark
except ImportError:
    benchmark = None  # type: ignore[assignment]
from utils import init_logging

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, LiberoEnv
try:
    from lerobot.envs.configs import MetaworldEnv
except ImportError:
    MetaworldEnv = None  # type: ignore[assignment]
from lerobot.utils.constants import OBS_ENV_STATE

try:
    from lerobot.envs.metaworld import DIFFICULTY_TO_TASKS as METAWORLD_TASK_GROUPS
except Exception:  # pragma: no cover - optional dependency/runtime
    METAWORLD_TASK_GROUPS = {}

from smolvla_qchunk.eval import (
    PolicyEvalBundle,
    build_policy_eval_bundle,
    evaluate_policy_with_best_of_n,
)


@EnvConfig.register_subclass("libero_compat")
@dataclass
class LiberoEnvCompat(LiberoEnv):
    selected_task_ids: tuple[int, ...] | None = None

    @property
    def gym_kwargs(self) -> dict:
        kwargs = dict(super().gym_kwargs)
        if self.selected_task_ids is not None:
            kwargs["task_ids"] = list(self.selected_task_ids)
        return kwargs


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy with optional critic augmentation")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH, help="Policy checkpoint directory (pretrained_model)")
    parser.add_argument("--critic-state", type=Path, default=DEFAULT_CRITIC_PATH, help="Path to critic_state.pt (optional).")
    parser.add_argument("--use-current-critic", type=_str2bool, default=True, help="Use current critic weights for eval (True/False).")
    parser.add_argument("--env-type", type=str, default="libero")
    env_task_group = parser.add_mutually_exclusive_group()
    env_task_group.add_argument(
        "--env-task",
        type=str,
        default="libero_object",
        help="Single LIBERO suite to evaluate (e.g., libero_object).",
    )
    env_task_group.add_argument(
        "--env-tasks",
        type=str,
        default= None,#["libero_object", "libero_spatial","libero_goal", "libero_10"]
        nargs="+",
        help="List of LIBERO suites to evaluate sequentially.",
    )
    parser.add_argument("--env-obs-type", type=str, default="pixels_agent_pos")
    parser.add_argument("--env-camera-name", type=str, default="agentview_image,robot0_eye_in_hand_image")
    parser.add_argument("--env-fps", type=int, default=30)
    parser.add_argument("--env-episode-length", type=int, default=520)
    parser.add_argument(
        "--env-task-ids",
        type=int,
        nargs="+",
        default=None,
        help="LIBERO task ids to evaluate (e.g. 0 1). Omit to run the first task only.",
    )
    parser.add_argument(
        "--suite-task-ids",
        type=str,
        nargs="+",
        default=None,#["libero_goal:0,1","libero_spatial:0,1"]
        help=(
            "Per-suite task ids formatted as suite:0,1. "
            "Example: --suite-task-ids libero_object:0,1 libero_goal:2,3"
        ),
    )
    parser.add_argument(
        "--eval-all-suite-tasks",
        action="store_true",
        help="Evaluate all tasks in each selected LIBERO suite (overridden when --suite-task-ids specifies that suite).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--best-of-n", type=int, default=2)
    parser.add_argument(
        "--use-best-of-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable critic-guided Best-of-N selection. Use --no-use-best-of-n to run plain SmolVLA.",
    )
    parser.add_argument(
        "--critic-q-agg-eval",
        type=str,
        default="min",
        choices=["mean", "min", "max"],
        help="Override critic q_aggregation during evaluation (defaults to the value stored in the checkpoint).",
    )
    parser.add_argument(
        "--use-data-augmentations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable visual data augmentations when encoding observations during eval.",
    )
    parser.add_argument(
        "--use-vlm-backbone-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass embeddings through the VLM backbone encode during eval.",
    )
    parser.add_argument("--max-render", type=int, default=1, help="Number of episodes to render/save.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory where eval artifacts (videos, eval_info.json) will be saved. Defaults to outputs/eval/bestofn_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=20)
    args = parser.parse_args()
    try:
        args.suite_task_ids = _parse_suite_task_ids(args.suite_task_ids)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    init_logging()
    if not args.use_best_of_n:
        logging.info("Best-of-N disabled; running plain SmolVLA policy.")
    critic_state = args.critic_state if args.use_best_of_n else None
    best_of_n = args.best_of_n if args.use_best_of_n else 1
    
    n_action_steps = args.n_action_steps
    if n_action_steps is None:
        logging.warning("Could not read policy n_action_steps from %s", args.policy_path)

    env_tasks = args.env_tasks if args.env_tasks is not None else [args.env_task]
    multi_task = len(env_tasks) > 1
    output_root = args.videos_dir or _default_eval_dir(
        use_best_of_n=args.use_best_of_n,
        best_of_n=best_of_n,
        n_action_steps=n_action_steps,
    )
    all_infos: dict[str, dict] = {}

    # Build policy/critic once to reuse across tasks.
    sample_task_ids = _task_ids_to_eval(args, env_tasks[0])
    if not sample_task_ids:
        raise ValueError(f"No task ids resolved for suite '{env_tasks[0]}'.")
    sample_task_id = sample_task_ids[0]
    sample_env_cfg = _build_env_config(args, task_name=env_tasks[0], task_ids=(sample_task_id,))
    policy_bundle: PolicyEvalBundle = build_policy_eval_bundle(
        policy_path=args.policy_path,
        critic_state_path=critic_state,
        sample_env_cfg=sample_env_cfg,
        sample_batch_size=1,
        device=args.device,
        best_of_n=best_of_n,
        n_action_steps=n_action_steps,
        critic_q_agg_override=args.critic_q_agg_eval,
        use_vlm_backbone_encode=args.use_vlm_backbone_encode,
        use_current_critic=args.use_current_critic,
    )

    for env_task in env_tasks:
        logging.info("Evaluating env task: %s", env_task)
        task_ids = _task_ids_to_eval(args, env_task)
        suite_output_root = output_root / env_task if multi_task else output_root
        suite_infos: list[dict[str, Any]] = []

        for task_id in task_ids:
            logging.info("  Task id: %s", task_id)
            env_cfg = _build_env_config(args, task_name=env_task, task_ids=(task_id,))
            videos_dir = suite_output_root / "videos" / f"task_{task_id}"
            videos_dir.mkdir(parents=True, exist_ok=True)

            info = evaluate_policy_with_best_of_n(
                policy_path=args.policy_path,
                critic_state_path=critic_state,
                env_cfg=env_cfg,
                n_episodes=args.n_episodes,
                batch_size=args.eval_batch_size,
                seed=args.seed,
                device=args.device,
                best_of_n=best_of_n,
                videos_dir=videos_dir,
                max_render=args.max_render,
                n_action_steps=n_action_steps,
                critic_q_agg_override=args.critic_q_agg_eval,
                policy_bundle=policy_bundle,
                use_vlm_backbone_encode=args.use_vlm_backbone_encode,
            )
            info["policy_path"] = str(args.policy_path)
            info["use_best_of_n"] = args.use_best_of_n
            info["env_task"] = env_task
            info["task_id"] = task_id
            info["use_vlm_backbone_encode"] = args.use_vlm_backbone_encode
            if args.use_best_of_n and critic_state is not None:
                info["critic_state_path"] = str(critic_state)
            _log_eval_summary(info)
            _save_eval_info(suite_output_root / f"task_{task_id}" / "eval_info.json", info)
            suite_infos.append(info)

        if suite_infos:
            merged = _merge_suite_infos(
                infos=suite_infos,
                env_task=env_task,
                policy_path=args.policy_path,
                use_best_of_n=args.use_best_of_n,
                best_of_n=best_of_n,
                n_action_steps=n_action_steps,
                critic_state=critic_state,
            )
            print(merged)
            _save_eval_info(suite_output_root / "eval_info.json", merged)
            all_infos[env_task] = merged

    if multi_task:
        _save_eval_info(output_root / "multi_eval_info.json", all_infos)

    if all_infos:
        summary = _build_summary(
            all_infos=all_infos,
            policy_path=args.policy_path,
            use_best_of_n=args.use_best_of_n,
            best_of_n=best_of_n,
            n_action_steps=n_action_steps,
            critic_state=critic_state,
        )
        _save_eval_info(output_root / "summary.json", summary)


def _build_env_config(
    args: argparse.Namespace, task_name: str | None = None, task_ids: tuple[int, ...] | None = None
) -> EnvConfig:
    task = task_name if task_name is not None else args.env_task
    if args.env_type == "libero":
        suite_task_ids = getattr(args, "suite_task_ids", {}) or {}
        if task_ids is None:
            suite_ids = None if task_name is None else suite_task_ids.get(task_name)
            if suite_ids is not None:
                task_ids = tuple(suite_ids)
            elif getattr(args, "eval_all_suite_tasks", False):
                task_ids = None
            else:
                task_ids = tuple(args.env_task_ids) if args.env_task_ids is not None else (0,)
        return LiberoEnvCompat(
            task=task,
            fps=args.env_fps,
            episode_length=args.env_episode_length,
            obs_type=args.env_obs_type,
            camera_name=args.env_camera_name,
            selected_task_ids=task_ids,
        )

    if args.env_type == "metaworld":
        if MetaworldEnv is None:
            raise ValueError(
                "env_type=metaworld was requested, but MetaworldEnv is unavailable in the current lerobot install."
            )
        # Default behavior: keep task groups (easy/medium/hard/very_hard) intact.
        # Only resolve to a concrete task when explicit task-ids are requested.
        explicit_task_id_mode = args.env_task_ids is not None or getattr(args, "eval_all_suite_tasks", False)
        if explicit_task_id_mode and task_ids is not None:
            if len(task_ids) != 1:
                raise ValueError(f"Expected exactly one metaworld task id, got {task_ids}.")
            requested_id = task_ids[0]
            tasks_in_group = METAWORLD_TASK_GROUPS.get(task)
            if tasks_in_group is not None:
                if requested_id < 0 or requested_id >= len(tasks_in_group):
                    raise ValueError(
                        f"Invalid MetaWorld task id {requested_id} for group '{task}'. "
                        f"Valid range: [0, {len(tasks_in_group) - 1}]"
                    )
                task = tasks_in_group[requested_id]
            elif requested_id != 0:
                raise ValueError(
                    f"Task '{task}' is not a known MetaWorld group; only task id 0 is valid for custom task names."
                )
        cfg = MetaworldEnv(
            task=task,
            fps=args.env_fps,
            episode_length=args.env_episode_length,
            obs_type=args.env_obs_type,
        )
        # Some MetaWorld policies expect both observation.state and observation.environment_state.
        # The default MetaworldEnv exposes only agent_pos -> observation.state; add a compatible
        # env-state feature declaration so policy/env feature checks can pass.
        if "environment_state" not in cfg.features:
            state_shape = cfg.features.get("agent_pos").shape if "agent_pos" in cfg.features else (4,)
            cfg.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=state_shape)
        if "environment_state" not in cfg.features_map:
            cfg.features_map["environment_state"] = OBS_ENV_STATE
        return cfg

    raise ValueError(f"Unsupported env_type '{args.env_type}'. Expected one of: libero, metaworld.")


def _log_eval_summary(info: dict) -> None:
    overall = info.get("overall", {})
    logging.info("Overall aggregated metrics:")
    for key, value in overall.items():
        if key == "video_paths":
            continue
        logging.info("  %s: %s", key, value)


def _save_eval_info(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _read_n_action_steps(policy_path: Path) -> int | None:
    config_path = policy_path / CONFIG_NAME
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None
    value = config.get("n_action_steps")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_ids_to_eval(args: argparse.Namespace, task_name: str) -> list[int]:
    if args.env_type == "metaworld":
        if args.env_task_ids is not None:
            return list(args.env_task_ids)
        if getattr(args, "eval_all_suite_tasks", False):
            tasks = METAWORLD_TASK_GROUPS.get(task_name)
            if tasks is not None:
                return list(range(len(tasks)))
        # Default: evaluate this group/task once (the underlying env may already be multi-task).
        return [0]

    suite_task_ids = getattr(args, "suite_task_ids", {}) or {}
    if task_name in suite_task_ids:
        return list(suite_task_ids[task_name])
    if getattr(args, "eval_all_suite_tasks", False):
        if benchmark is None:
            raise ValueError(
                "env_type=libero requires the `libero` package when --eval-all-suite-tasks is enabled."
            )
        bench = benchmark.get_benchmark_dict()
        if task_name not in bench:
            raise ValueError(f"Unknown LIBERO suite '{task_name}'. Available: {', '.join(sorted(bench.keys()))}")
        suite = bench[task_name]()
        return list(range(len(suite.tasks)))
    if args.env_task_ids is not None:
        return list(args.env_task_ids)
    return [0]


def _merge_suite_infos(
    *,
    infos: list[dict[str, Any]],
    env_task: str,
    policy_path: Path,
    use_best_of_n: bool,
    best_of_n: int,
    n_action_steps: int | None,
    critic_state: Path | None,
) -> dict[str, Any]:
    drop_keys = {"video_paths", "sum_rewards", "max_rewards"}

    per_task: list[dict[str, Any]] = []
    per_group_accumulator: dict[str, dict[str, Any]] = {}
    total_episodes = 0
    total_sum_reward = 0.0
    total_max_reward = 0.0
    total_success = 0.0
    total_eval_time = 0.0
    total_eval_ep_time = 0.0

    for info in infos:
        for task_entry in info.get("per_task", []):
            metrics = {
                key: value for key, value in task_entry.get("metrics", {}).items() if key not in drop_keys
            }
            per_task.append(
                {
                    "task_group": task_entry.get("task_group", env_task),
                    "task_id": task_entry.get("task_id"),
                    "metrics": metrics,
                }
            )

        for group_name, group_metrics in info.get("per_group", {}).items():
            accumulator = per_group_accumulator.setdefault(
                group_name,
                {
                    "sum_reward_total": 0.0,
                    "max_reward_total": 0.0,
                    "success_total": 0.0,
                    "episodes": 0,
                },
            )
            n_eps = group_metrics.get("n_episodes", 0)
            accumulator["sum_reward_total"] += group_metrics.get("avg_sum_reward", 0.0) * n_eps
            accumulator["max_reward_total"] += group_metrics.get("avg_max_reward", 0.0) * n_eps
            accumulator["success_total"] += group_metrics.get("pc_success", 0.0) / 100.0 * n_eps
            accumulator["episodes"] += n_eps

        overall = info.get("overall", {})
        n_eps = overall.get("n_episodes", 0)
        total_episodes += n_eps
        total_sum_reward += overall.get("avg_sum_reward", 0.0) * n_eps
        total_max_reward += overall.get("avg_max_reward", 0.0) * n_eps
        total_success += overall.get("pc_success", 0.0) / 100.0 * n_eps
        total_eval_time += overall.get("eval_s", 0.0)
        total_eval_ep_time += overall.get("eval_ep_s", 0.0) * n_eps

    per_group: dict[str, dict[str, Any]] = {}
    for name, aggregates in per_group_accumulator.items():
        n_eps = aggregates["episodes"]
        if n_eps == 0:
            continue
        per_group[name] = {
            "avg_sum_reward": aggregates["sum_reward_total"] / n_eps,
            "avg_max_reward": aggregates["max_reward_total"] / n_eps,
            "pc_success": 100.0 * aggregates["success_total"] / n_eps,
            "n_episodes": n_eps,
        }

    overall_summary: dict[str, Any] = {"n_episodes": total_episodes}
    if total_episodes > 0:
        overall_summary["avg_sum_reward"] = total_sum_reward / total_episodes
        overall_summary["avg_max_reward"] = total_max_reward / total_episodes
        overall_summary["pc_success"] = 100.0 * total_success / total_episodes
        overall_summary["eval_ep_s"] = total_eval_ep_time / total_episodes if total_eval_ep_time > 0 else 0.0
    overall_summary["eval_s"] = total_eval_time

    merged: dict[str, Any] = {
        "per_task": per_task,
        "per_group": per_group,
        "overall": overall_summary,
        "env_task": env_task,
        "policy_path": str(policy_path),
        "use_best_of_n": use_best_of_n,
        "best_of_n": best_of_n,
        "n_action_steps": n_action_steps,
        "task_ids": [info.get("task_id") for info in infos if "task_id" in info],
    }

    if use_best_of_n and critic_state is not None:
        merged["critic_state_path"] = str(critic_state)

    return merged


def _default_eval_dir(
    *, use_best_of_n: bool, best_of_n: int, n_action_steps: int | None
) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    best_of_n_suffix = f"bestofn{best_of_n}"
    critic_suffix = "critic" if use_best_of_n and best_of_n > 1 else "no-critic"
    if n_action_steps is None:
        policy_suffix = "nstep-unknown"
    else:
        policy_suffix = f"nstep{n_action_steps}"
    suffixes = [best_of_n_suffix, critic_suffix, policy_suffix]
    label = "_".join(suffixes)
    return REPO_ROOT / "outputs" / "eval" / f"{stamp}_{label}"


def _parse_suite_task_ids(entries: list[str] | None) -> dict[str, tuple[int, ...]]:
    if not entries:
        return {}
    mapping: dict[str, tuple[int, ...]] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid suite-task-ids entry '{entry}'. Expected format suite:id0,id1")
        suite, ids_str = entry.split(":", 1)
        suite = suite.strip()
        if not suite:
            raise ValueError(f"Suite name missing in entry '{entry}'")
        try:
            ids = tuple(int(part.strip()) for part in ids_str.split(",") if part.strip() != "")
        except ValueError as exc:
            raise ValueError(f"Invalid task id in entry '{entry}': {exc}") from exc
        if not ids:
            raise ValueError(f"No task ids specified for suite '{suite}' in entry '{entry}'")
        mapping[suite] = ids
    return mapping


def _build_summary(
    *,
    all_infos: dict[str, dict],
    policy_path: Path,
    use_best_of_n: bool,
    best_of_n: int,
    n_action_steps: int | None,
    critic_state: Path | None,
) -> dict[str, Any]:
    drop_keys = {"video_paths", "sum_rewards", "max_rewards"}

    task_success: dict[str, dict[str, float]] = {}
    per_group_accumulator: dict[str, dict[str, Any]] = {}
    total_episodes = 0
    total_sum_reward = 0.0
    total_max_reward = 0.0
    total_success = 0.0
    total_eval_time = 0.0
    total_eval_ep_time = 0.0

    for env_task, info in all_infos.items():
        for task_entry in info.get("per_task", []):
            metrics = {
                key: value
                for key, value in task_entry.get("metrics", {}).items()
                if key not in drop_keys
            }
            # 直接记录每个 task 的成功率
            success_rate: float | None = None
            if "successes" in metrics and metrics["successes"]:
                successes = metrics["successes"]
                success_rate = 100.0 * (sum(successes) / len(successes))
            elif "pc_success" in metrics:
                success_rate = float(metrics["pc_success"])
            if success_rate is not None:
                task_success.setdefault(env_task, {})[str(task_entry.get("task_id"))] = success_rate

        for group_name, group_metrics in info.get("per_group", {}).items():
            accumulator = per_group_accumulator.setdefault(
                group_name,
                {
                    "sum_reward_total": 0.0,
                    "max_reward_total": 0.0,
                    "success_total": 0.0,
                    "episodes": 0,
                },
            )
            n_eps = group_metrics.get("n_episodes", 0)
            accumulator["sum_reward_total"] += group_metrics.get("avg_sum_reward", 0.0) * n_eps
            accumulator["max_reward_total"] += group_metrics.get("avg_max_reward", 0.0) * n_eps
            accumulator["success_total"] += group_metrics.get("pc_success", 0.0) / 100.0 * n_eps
            accumulator["episodes"] += n_eps

        overall = info.get("overall", {})
        n_eps = overall.get("n_episodes", 0)
        total_episodes += n_eps
        total_sum_reward += overall.get("avg_sum_reward", 0.0) * n_eps
        total_max_reward += overall.get("avg_max_reward", 0.0) * n_eps
        total_success += overall.get("pc_success", 0.0) / 100.0 * n_eps
        total_eval_time += overall.get("eval_s", 0.0)
        total_eval_ep_time += overall.get("eval_ep_s", 0.0) * n_eps

    per_group: dict[str, dict[str, Any]] = {}
    for name, aggregates in per_group_accumulator.items():
        n_eps = aggregates["episodes"]
        if n_eps == 0:
            continue
        per_group[name] = {
            "avg_sum_reward": aggregates["sum_reward_total"] / n_eps,
            "avg_max_reward": aggregates["max_reward_total"] / n_eps,
            "pc_success": 100.0 * aggregates["success_total"] / n_eps,
            "n_episodes": n_eps,
        }

    overall_summary: dict[str, Any] = {"n_episodes": total_episodes}
    if total_episodes > 0:
        overall_summary["avg_sum_reward"] = total_sum_reward / total_episodes
        overall_summary["avg_max_reward"] = total_max_reward / total_episodes
        overall_summary["pc_success"] = 100.0 * total_success / total_episodes
        overall_summary["eval_ep_s"] = total_eval_ep_time / total_episodes if total_eval_ep_time > 0 else 0.0
    overall_summary["eval_s"] = total_eval_time

    summary: dict[str, Any] = {
        "task_success": task_success,
        "per_group": per_group,
        "overall": overall_summary,
        "env_tasks": list(all_infos.keys()),
        "policy_path": str(policy_path),
        "use_best_of_n": use_best_of_n,
        "best_of_n": best_of_n,
        "n_action_steps": n_action_steps,
    }

    if use_best_of_n and critic_state is not None:
        summary["critic_state_path"] = str(critic_state)

    return summary


if __name__ == "__main__":
    main()
