#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
JOB_NAME="${JOB_NAME:-eval_metaworld}"
LOG_ROOT="${LOG_ROOT:-test_log}"
NOHUP_LOG_DIR="${NOHUP_LOG_DIR:-logs/${JOB_NAME}}"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  mkdir -p "${NOHUP_LOG_DIR}"
  NOHUP_LOG_FILE="${NOHUP_LOG_DIR}/${JOB_NAME}_$(date +%F_%H-%M-%S).log"
  NOHUP_LOG_FILE_ABS="$(readlink -f "${NOHUP_LOG_FILE}")"
  nohup bash "$0" "$@" >"${NOHUP_LOG_FILE_ABS}" 2>&1 &
  echo "started in background: pid=$! log= ${NOHUP_LOG_FILE_ABS}"
  exit 0
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LEROBOT_MAX_EPISODES_RENDERED="${LEROBOT_MAX_EPISODES_RENDERED:-0}"

CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate VGAS

# --------------------------- Editable Config --------------------------- #
CKPT_ROOT_REL="${CKPT_ROOT_REL:-outputs/YOUR_METAWORLD_BC_RUN/checkpoints}"
STEPS_STR="${STEPS_STR:-40000 15000}"
TASK_GROUPS_STR="${TASK_GROUPS_STR:-very_hard}"  # example: very_hard (set "easy medium hard very_hard" for all)
SEEDS_STR="${SEEDS_STR:-0 42 1234 124 410 2000}"
read -r -a STEPS <<< "${STEPS_STR}"
read -r -a TASK_GROUPS <<< "${TASK_GROUPS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-20}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-20}"
DEVICE="${DEVICE:-cuda}"
# Override policy n_action_steps at eval time (must be <= chunk_size in policy config).
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
RENAME_MAP="${RENAME_MAP:-{\"observation.image\":\"observation.image\"}}"
# ---------------------------------------------------------------------- #

CKPT_ROOT="${REPO_ROOT}/${CKPT_ROOT_REL}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output_metaworld/BC_Only_5_SHOT_N_ACTION_STEP_20_${EVAL_N_EPISODES}_episodes_new}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root does not exist: ${CKPT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "Repo root:        ${REPO_ROOT}"
echo "Checkpoint root:  ${CKPT_ROOT}"
echo "Task groups:      ${TASK_GROUPS[*]}"
echo "Steps:            ${STEPS[*]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Output root:      ${OUT_ROOT}"

write_summary_json() {
  local eval_info_path="$1"
  local summary_path="$2"
  local policy_path="$3"
  local step_dir="$4"
  local seed="$5"
  local task_group="$6"
  local out_dir="$7"

  python3 - "${eval_info_path}" "${summary_path}" "${policy_path}" "${step_dir}" "${seed}" "${task_group}" "${out_dir}" "${N_ACTION_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

eval_info_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
policy_path = sys.argv[3]
step_dir = sys.argv[4]
seed_raw = sys.argv[5]
task_group = sys.argv[6]
out_dir = sys.argv[7]
n_action_steps_raw = sys.argv[8]

try:
    seed = int(seed_raw)
except ValueError:
    seed = seed_raw

try:
    n_action_steps = int(n_action_steps_raw)
except ValueError:
    n_action_steps = n_action_steps_raw

with eval_info_path.open("r", encoding="utf-8") as f:
    info = json.load(f)

per_task = info.get("per_task", [])
task_success = {task_group: {}}
per_group = {}

for task_entry in per_task:
    task_id = task_entry.get("task_id")
    task_id_key = str(task_id)
    metrics = task_entry.get("metrics", {})

    sum_rewards = metrics.get("sum_rewards") or []
    max_rewards = metrics.get("max_rewards") or []
    successes = metrics.get("successes") or []

    success_values = [1.0 if bool(x) else 0.0 for x in successes]
    success_rate = 100.0 * sum(success_values) / len(success_values) if success_values else 0.0
    task_success[task_group][task_id_key] = success_rate

    n_episodes = len(successes) if successes else max(len(sum_rewards), len(max_rewards))
    avg_sum_reward = sum(sum_rewards) / len(sum_rewards) if sum_rewards else 0.0
    avg_max_reward = sum(max_rewards) / len(max_rewards) if max_rewards else 0.0
    group_name = f"task_{task_id_key}"
    per_group[group_name] = {
        "avg_sum_reward": avg_sum_reward,
        "avg_max_reward": avg_max_reward,
        "pc_success": success_rate,
        "n_episodes": n_episodes,
    }

if not per_group:
    for group_name, group_metrics in info.get("per_group", {}).items():
        per_group[group_name] = {
            "avg_sum_reward": float(group_metrics.get("avg_sum_reward", 0.0)),
            "avg_max_reward": float(group_metrics.get("avg_max_reward", 0.0)),
            "pc_success": float(group_metrics.get("pc_success", 0.0)),
            "n_episodes": int(group_metrics.get("n_episodes", 0)),
        }

overall = info.get("overall", {})
summary = {
    "task_success": task_success,
    "per_group": per_group,
    "overall": overall,
    "env_tasks": [task_group],
    "policy_path": policy_path,
    "n_action_steps": n_action_steps,
    "step": step_dir,
    "seed": seed,
    "task_group": task_group,
    "output_dir": out_dir,
    "eval_info_path": str(eval_info_path),
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
PY
}

for seed in "${SEEDS[@]}"; do
  for step_raw in "${STEPS[@]}"; do
    if [[ "${step_raw}" =~ ^[0-9]+$ ]]; then
      step_dir="$(printf "%06d" "$((10#${step_raw}))")"
    else
      step_dir="${step_raw}"
    fi

    POLICY_PATH="${CKPT_ROOT}/${step_dir}/pretrained_model"
    if [[ ! -d "${POLICY_PATH}" ]]; then
      echo "[skip] missing policy path: ${POLICY_PATH}" >&2
      continue
    fi

    for task_group in "${TASK_GROUPS[@]}"; do
      OUT_DIR="${OUT_ROOT}/${task_group}/step_${step_dir}/seed_${seed}"
      EVAL_INFO_PATH="${OUT_DIR}/eval_info.json"
      SUMMARY_PATH="${OUT_DIR}/summary.json"
      if [[ -f "${EVAL_INFO_PATH}" ]]; then
        if [[ ! -f "${SUMMARY_PATH}" ]]; then
          echo "[info] found ${EVAL_INFO_PATH}, generating ${SUMMARY_PATH}"
          write_summary_json "${EVAL_INFO_PATH}" "${SUMMARY_PATH}" "${POLICY_PATH}" "${step_dir}" "${seed}" "${task_group}" "${OUT_DIR}"
        else
          echo "[skip] found ${SUMMARY_PATH}"
        fi
        continue
      fi

      if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
        echo "[skip] SUMMARY_ONLY=1 and missing ${EVAL_INFO_PATH}"
        continue
      fi

      mkdir -p "${OUT_DIR}"

      LOG_DIR="${LOG_ROOT}/${JOB_NAME}/${task_group}/step_${step_dir}/seed_${seed}"
      mkdir -p "${LOG_DIR}"
      LOG_FILE="${LOG_DIR}/eval.log"

      echo "[run] task_group=${task_group} seed=${seed} step=${step_dir}"
      echo "      policy= ${POLICY_PATH}"
      echo "      output= ${OUT_DIR}"
      echo "      log= ${LOG_FILE}"

      lerobot-eval \
        --policy.path="${POLICY_PATH}" \
        --policy.device="${DEVICE}" \
        --policy.n_action_steps="${N_ACTION_STEPS}" \
        --env.type=metaworld \
        --env.task="${task_group}" \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.n_episodes="${EVAL_N_EPISODES}" \
        --seed="${seed}" \
        --rename_map="${RENAME_MAP}" \
        --output_dir="${OUT_DIR}" \
        >"${LOG_FILE}" 2>&1 &

      PID=$!
      status=0
      wait "${PID}" || status=$?
      if [[ "${status}" -ne 0 ]]; then
        echo "[error] lerobot-eval exited with status ${status} (log= ${LOG_FILE})"
        tail -n 50 "${LOG_FILE}" || true
      else
        EVAL_INFO_PATH="${OUT_DIR}/eval_info.json"
        SUMMARY_PATH="${OUT_DIR}/summary.json"
        if [[ -f "${EVAL_INFO_PATH}" ]]; then
          write_summary_json "${EVAL_INFO_PATH}" "${SUMMARY_PATH}" "${POLICY_PATH}" "${step_dir}" "${seed}" "${task_group}" "${OUT_DIR}"
          echo "[done] summary= ${SUMMARY_PATH}"
        else
          echo "[warn] missing ${EVAL_INFO_PATH}; skip summary generation" >&2
        fi
      fi
    done
  done
done
