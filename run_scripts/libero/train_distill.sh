#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

JOB_NAME="${JOB_NAME:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate VGAS

SHOT_LABEL="${SHOT_LABEL:-5_SHOT}"
DATASET_NAME="${DATASET_NAME:-libero_goal}"

POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/pretrained_vla/Libero/smolvla/${SHOT_LABEL}/pretrained_model}"
TEACHER_POLICY_PATH="${TEACHER_POLICY_PATH:-${POLICY_PATH}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/Libero/HF_LIBERO_${SHOT_LABEL}/${DATASET_NAME}}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/train/smolvla_distill/${SHOT_LABEL}/${DATASET_NAME}}"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"


######################################################
STUDENT_NUM_STEPS="${STUDENT_NUM_STEPS:-8}"
DISTILL_PHASE1_STEPS="${DISTILL_PHASE1_STEPS:-2000}"
PHASE1_TRAIN_STUDENT="${PHASE1_TRAIN_STUDENT:-false}"
PHASE1_USE_STUDENT_AS_SOURCE="${PHASE1_USE_STUDENT_AS_SOURCE:-true}"
PHASE1_CRITIC_UPDATES_PER_STEP="${PHASE1_CRITIC_UPDATES_PER_STEP:-3}"
PHASE1_TEACHER_ACTION_SAMPLES="${PHASE1_TEACHER_ACTION_SAMPLES:-1}"
PHASE2_GLOBAL_SAMPLES="${PHASE2_GLOBAL_SAMPLES:-4}"
PHASE2_LOCAL_SAMPLES="${PHASE2_LOCAL_SAMPLES:-2}"
PHASE2_LOCAL_OPT_STEPS="${PHASE2_LOCAL_OPT_STEPS:-5}"
PHASE2_LOCAL_OPT_LR="${PHASE2_LOCAL_OPT_LR:-3e-4}"
PHASE2_LOCAL_GRAD_NORMALIZE="${PHASE2_LOCAL_GRAD_NORMALIZE:-true}"
PHASE2_LOCAL_ADV_WEIGHT="${PHASE2_LOCAL_ADV_WEIGHT:-false}"
PHASE2_LOCAL_ADV_EPS="${PHASE2_LOCAL_ADV_EPS:-1e-6}"
PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL="${PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL:-true}"
PHASE2_USE_TEACHER_ANCHOR="${PHASE2_USE_TEACHER_ANCHOR:-false}"
MASK_PADDED_ACTION_LOSS="${MASK_PADDED_ACTION_LOSS:-true}"
DISTILL_LAMBDA_W2="${DISTILL_LAMBDA_W2:-0.2}"
DISTILL_LAMBDA_OPT="${DISTILL_LAMBDA_OPT:-1.0}"

######################################################
CRITIC_BACKUP_ACTION_SAMPLES="${CRITIC_BACKUP_ACTION_SAMPLES:-${CRITIC_ACTION_SAMPLES:-4}}"
CRITIC_UPDATES_PER_STEP="${CRITIC_UPDATES_PER_STEP:-3}"
CRITIC_VISION_FREEZE_LAYERS="${CRITIC_VISION_FREEZE_LAYERS:-11}"
OOD_M_ACTIONS="${OOD_M_ACTIONS:-4}"
CRITIC_USE_STATE_ENCODER="${CRITIC_USE_STATE_ENCODER:-true}"
CRITIC_USE_INDEPENDENT_ENCODER="${CRITIC_USE_INDEPENDENT_ENCODER:-true}"
USE_VLM_BACKBONE_ENCODE="${USE_VLM_BACKBONE_ENCODE:-true}"
TRAIN_FULL_MODEL="${TRAIN_FULL_MODEL:-false}"
UNFREEZE_VISION_ENCODER="${UNFREEZE_VISION_ENCODER:-false}"
ENV_TASK="${ENV_TASK:-libero_goal}"
if [[ -z "${JOB_NAME}" ]]; then
  DATASET_TAG="${DATASET_NAME#libero_}"
  if [[ -z "${DATASET_TAG}" ]]; then
    DATASET_TAG="${ENV_TASK#libero_}"
  fi
  if [[ -z "${DATASET_TAG}" ]]; then
    DATASET_TAG="${DATASET_NAME}"
  fi
  CORE_TAG="d2_${SHOT_LABEL}_${DATASET_TAG}_s${STUDENT_NUM_STEPS}_p1stu$([[ "${PHASE1_TRAIN_STUDENT}" == "true" ]] && echo 1 || echo 0)_ta$([[ "${PHASE2_USE_TEACHER_ANCHOR}" == "true" ]] && echo 1 || echo 0)_cse$([[ "${CRITIC_USE_STATE_ENCODER}" == "true" ]] && echo 1 || echo 0)_g${PHASE2_GLOBAL_SAMPLES}m${PHASE2_LOCAL_SAMPLES}_aw$([[ "${PHASE2_LOCAL_ADV_WEIGHT}" == "true" ]] && echo 1 || echo 0)_gn$([[ "${PHASE2_LOCAL_GRAD_NORMALIZE}" == "true" ]] && echo 1 || echo 0)_da$([[ "${PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL}" == "true" ]] && echo 1 || echo 0)_mp$([[ "${MASK_PADDED_ACTION_LOSS}" == "true" ]] && echo 1 || echo 0)"
  KEY_TAG="p1${DISTILL_PHASE1_STEPS}_ls${PHASE2_LOCAL_OPT_STEPS}_lr${PHASE2_LOCAL_OPT_LR}_eps${PHASE2_LOCAL_ADV_EPS}_k${CRITIC_BACKUP_ACTION_SAMPLES}_seed${SEED}"
  SHORT_HASH="$(printf '%s' "${CORE_TAG}_${KEY_TAG}" | md5sum | cut -c1-6)"
  JOB_NAME="${CORE_TAG}_${SHORT_HASH}_$(date +%m%d_%H%M)"
fi

LOG_DIR="${LOG_DIR:-logs/${JOB_NAME}}"
FINAL_OUTPUT_DIR="${OUTPUT_BASE}/${JOB_NAME}"
PYTHON_PID_FILE="${PYTHON_PID_FILE:-}"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%F_%H-%M-%S).log"
  LOG_FILE_ABS="$(readlink -f "${LOG_FILE}")"
  PYTHON_PID_FILE_RUN="${LOG_DIR}/${JOB_NAME}.python.pid"
  rm -f "${PYTHON_PID_FILE_RUN}"
  nohup env JOB_NAME="${JOB_NAME}" LOG_DIR="${LOG_DIR}" PYTHON_PID_FILE="${PYTHON_PID_FILE_RUN}" bash "$0" "$@" >"${LOG_FILE_ABS}" 2>&1 &
  SHELL_PID=$!
  PYTHON_PID=""
  for _ in {1..50}; do
    if [[ -s "${PYTHON_PID_FILE_RUN}" ]]; then
      PYTHON_PID="$(cat "${PYTHON_PID_FILE_RUN}" 2>/dev/null || true)"
      break
    fi
    sleep 0.1
  done
  echo "started in background: shell_pid=${SHELL_PID} python_pid=${PYTHON_PID:-pending} output_dir=${FINAL_OUTPUT_DIR} log= ${LOG_FILE_ABS}"
  exit 0
fi

EXTRA_FLAGS=()
if [[ "${PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL}" == "true" ]]; then
  EXTRA_FLAGS+=(--phase2-include-dataset-action-in-global-pool)
fi
if [[ "${TRAIN_FULL_MODEL}" == "true" ]]; then
  EXTRA_FLAGS+=(--train-full-model)
fi
if [[ "${UNFREEZE_VISION_ENCODER}" == "true" ]]; then
  EXTRA_FLAGS+=(--unfreeze-vision-encoder)
fi

if [[ "${WANDB_ENABLE:-true}" == "true" ]]; then
  EXTRA_FLAGS+=(
    --wandb
    --wandb-mode="${WANDB_MODE:-online}"
    --wandb-project="${WANDB_PROJECT:-my_smolvla_distill}"
  )
fi
echo "[run] JOB_NAME=${JOB_NAME}"
echo "[run] OUTPUT_DIR=${FINAL_OUTPUT_DIR}"

TRAIN_CMD=(
  python scripts/run_qchunk_offline.py
  --distill
  "--policy-path=${POLICY_PATH}"
  "--teacher-policy-path=${TEACHER_POLICY_PATH}"
  "--dataset-root=${DATASET_ROOT}"
  "--dataset-repo-id=${DATASET_NAME}"
  "--output-dir=${OUTPUT_BASE}"
  --output-dir-layout=tag
  "--job-name=${JOB_NAME}"
  "--steps=${STEPS}"
  "--batch-size=${BATCH_SIZE}"
  "--log-interval=${LOG_INTERVAL}"
  "--seed=${SEED}"
  --eval-freq=0
  --chunk-size=32
  --n-action-steps=20
  --q-chunk-len=32
  --critic-type=q_chunk_former
  --critic-lr=1e-4
  --critic-lr-final=2.5e-5
  "--critic-total-steps=${STEPS}"
  --critic-q-agg=min
  --critic-loss-mode=per_head_mean
  --critic-att-mode=bi-level
  "--critic-use-state-encoder=${CRITIC_USE_STATE_ENCODER}"
  "--critic-use-independent-encoder=${CRITIC_USE_INDEPENDENT_ENCODER}"
  "--critic-vision-freeze-layers=${CRITIC_VISION_FREEZE_LAYERS}"
  "--use-vlm-backbone-encode=${USE_VLM_BACKBONE_ENCODE}"
  "--critic-updates-per-step=${CRITIC_UPDATES_PER_STEP}"
  --critic-grad-clip=10
  "--critic-backup-action-samples=${CRITIC_BACKUP_ACTION_SAMPLES}"
  "--ood-m-actions=${OOD_M_ACTIONS}"
  --use-calql=false
  --use-ood-reg=true
  --ood-alpha=5.0
  --dist-penalty-beta=5.0
  --use-ood-noise=true
  --use-ood-trunc=false
  --use-ood-mix=true
  --ood-noise-stds 0.02
  --ood-mix-ratio=1.0
  --ood-mix-alpha-low=0.2
  --ood-mix-alpha-high=0.8
  --loss-rank-weight=5.0
  "--student-num-steps=${STUDENT_NUM_STEPS}"
  "--distill-phase1-steps=${DISTILL_PHASE1_STEPS}"
  "--phase1-teacher-action-samples=${PHASE1_TEACHER_ACTION_SAMPLES}"
  "--phase1-train-student=${PHASE1_TRAIN_STUDENT}"
  "--phase1-use-student-as-source=${PHASE1_USE_STUDENT_AS_SOURCE}"
  ${PHASE1_CRITIC_UPDATES_PER_STEP:+"--phase1-critic-updates-per-step=${PHASE1_CRITIC_UPDATES_PER_STEP}"}
  "--phase2-global-samples=${PHASE2_GLOBAL_SAMPLES}"
  "--phase2-local-samples=${PHASE2_LOCAL_SAMPLES}"
  "--phase2-local-opt-steps=${PHASE2_LOCAL_OPT_STEPS}"
  "--phase2-local-opt-lr=${PHASE2_LOCAL_OPT_LR}"
  "--phase2-local-grad-normalize=${PHASE2_LOCAL_GRAD_NORMALIZE}"
  "--phase2-local-adv-weight=${PHASE2_LOCAL_ADV_WEIGHT}"
  "--phase2-local-adv-eps=${PHASE2_LOCAL_ADV_EPS}"
  "--phase2-use-teacher-anchor=${PHASE2_USE_TEACHER_ANCHOR}"
  "--mask-padded-action-loss=${MASK_PADDED_ACTION_LOSS}"
  "--distill-lambda-w2=${DISTILL_LAMBDA_W2}"
  "--distill-lambda-opt=${DISTILL_LAMBDA_OPT}"
  --phase2-keep-critic-best-of-n
  --env-type=libero
  "--env-task=${ENV_TASK}"
  "${EXTRA_FLAGS[@]}"
)

if [[ -n "${PYTHON_PID_FILE}" ]]; then
  "${TRAIN_CMD[@]}" &
  PYTHON_PID=$!
  echo "${PYTHON_PID}" > "${PYTHON_PID_FILE}"
  echo "[run] PYTHON_PID=${PYTHON_PID}"
  wait "${PYTHON_PID}"
else
  "${TRAIN_CMD[@]}"
fi
