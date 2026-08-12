#!/usr/bin/env bash
# Evaluate a frozen LIBERO SFT policy without a critic or Best-of-N reranking.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"

CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate VGAS

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/LIBERO}"
if [ -d "${LIBERO_ROOT}" ]; then
  export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
fi

ENV_TASK="${ENV_TASK:-libero_goal}"
SHOT_LABEL="${SHOT_LABEL:-5_SHOT}"
N_ACTION="${N_ACTION:-20}"
EVAL_BATCH="${EVAL_BATCH:-50}"
N_EPISODES="${N_EPISODES:-${EVAL_BATCH}}"
read -r -a SEEDS <<< "${SEEDS:-0 42 1234 124 410 2000}"

BASE_OUT="${BASE_OUT:-${REPO_ROOT}/outputs/eval}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/test_log}"
JOB_NAME="${JOB_NAME:-test_bc}"
POLICY_BASE_DIR="${POLICY_BASE_DIR:-${REPO_ROOT}/pretrained_vla/Libero/smolvla/${SHOT_LABEL}}"

# Example: STEP_LIST="10000 15000" bash run_scripts/libero/test_bc.sh
STEP_LIST_STR="${STEP_LIST:-}"
if [ -n "${STEP_LIST_STR}" ]; then
  read -r -a STEP_LIST <<< "${STEP_LIST_STR}"
else
  STEP_LIST=("base")
fi

TASK_ID="${TASK_ID:-}"
EVAL_ALL_SUITE_TASKS="${EVAL_ALL_SUITE_TASKS:-true}"
TASK_FLAGS=()
if [ "${EVAL_ALL_SUITE_TASKS}" = "true" ]; then
  TASK_FLAGS+=(--eval-all-suite-tasks)
elif [ -n "${TASK_ID}" ]; then
  TASK_FLAGS+=(--env-task-ids "${TASK_ID}")
else
  TASK_FLAGS+=(--env-task-ids 0)
fi

for STEP in "${STEP_LIST[@]}"; do
  if [ "${STEP}" = "base" ]; then
    STEP_TAG="base"
    POLICY_PATH="${POLICY_BASE_DIR}/pretrained_model"
  else
    STEP_TAG="step_${STEP}"
    POLICY_PATH="${POLICY_BASE_DIR}/${STEP}/pretrained_model"
  fi

  if [ ! -d "${POLICY_PATH}" ]; then
    echo "[error] policy path not found: ${POLICY_PATH}" >&2
    exit 1
  fi

  DIR_NAME="BC_${SHOT_LABEL}_${STEP_TAG}_EXEC_${N_ACTION}_BATCH${EVAL_BATCH}"
  JOB_NAME_STEP="${JOB_NAME}_${SHOT_LABEL}_${STEP_TAG}"

  for seed in "${SEEDS[@]}"; do
    OUT_DIR="${BASE_OUT}/${DIR_NAME}/BC_Only/${ENV_TASK}/seed_${seed}"
    SUMMARY_PATH="${OUT_DIR}/summary.json"
    if [ -f "${SUMMARY_PATH}" ]; then
      echo "[skip] found ${SUMMARY_PATH}"
      continue
    fi

    LOG_DIR="${LOG_ROOT}/${JOB_NAME_STEP}/BC_Only/seed_${seed}"
    LOG_FILE="${LOG_DIR}/eval.log"
    mkdir -p "${OUT_DIR}" "${LOG_DIR}"

    echo "[run] step=${STEP} seed=${seed} policy=${POLICY_PATH}"
    echo "[run] log=${LOG_FILE}"
    python scripts/eval_qc_bestofn.py \
      --env-task="${ENV_TASK}" \
      --policy-path="${POLICY_PATH}" \
      --videos-dir="${OUT_DIR}" \
      --seed "${seed}" \
      --eval-batch-size "${EVAL_BATCH}" \
      --n-action-steps "${N_ACTION}" \
      --n-episodes "${N_EPISODES}" \
      --no-use-best-of-n \
      --best-of-n 1 \
      "${TASK_FLAGS[@]}" \
      >"${LOG_FILE}" 2>&1 &
    PY_PID=$!
    echo "[pid] ${PY_PID}"

    status=0
    wait "${PY_PID}" || status=$?
    if [ "${status}" -ne 0 ]; then
      echo "[error] python exited with status ${status} (log=${LOG_FILE})" >&2
      tail -n 50 "${LOG_FILE}"
      exit "${status}"
    fi

    if [ -f "${SUMMARY_PATH}" ]; then
      python - "${SUMMARY_PATH}" "${seed}" <<'PY'
import json
import sys

path, seed = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict):
    data.setdefault("seed", seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
PY
    fi
  done
done
