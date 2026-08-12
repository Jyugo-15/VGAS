#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SHOT_LABEL="${SHOT_LABEL:-5_SHOT}"
TASK_SPLIT="${TASK_SPLIT:-very_hard}"
DATASET_REPO_ID="${DATASET_REPO_ID:-metaworld_local}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/pretrained_vla/MetaWorld/smolvla/${SHOT_LABEL}/pretrained_model}"
CRITIC_INIT_CHECKPOINT="${CRITIC_INIT_CHECKPOINT:-}"
TEACHER_POLICY_PATH="${TEACHER_POLICY_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/MetaWorld/MT50_${SHOT_LABEL}/${TASK_SPLIT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/train/smolvla_distill/metaworld/${SHOT_LABEL}/${TASK_SPLIT}}"

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
Q_CHUNK_LEN="${Q_CHUNK_LEN:-32}"
STUDENT_NUM_STEPS="${STUDENT_NUM_STEPS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
POLICY_LR="${POLICY_LR:-}"
POLICY_SCHEDULER_WARMUP_STEPS="${POLICY_SCHEDULER_WARMUP_STEPS:-}"
POLICY_SCHEDULER_DECAY_STEPS="${POLICY_SCHEDULER_DECAY_STEPS:-}"
POLICY_SCHEDULER_DECAY_LR="${POLICY_SCHEDULER_DECAY_LR:-}"

# Libero-style two-stage schedule: phase-1 critic warmup, then phase-2 actor+critic.
DISTILL_PHASE1_STEPS="${DISTILL_PHASE1_STEPS:-3000}"
PHASE1_TRAIN_STUDENT="${PHASE1_TRAIN_STUDENT:-false}"
PHASE1_USE_STUDENT_AS_SOURCE="${PHASE1_USE_STUDENT_AS_SOURCE:-true}"
PHASE1_CRITIC_UPDATES_PER_STEP="${PHASE1_CRITIC_UPDATES_PER_STEP:-2}"
PHASE1_TEACHER_ACTION_SAMPLES="${PHASE1_TEACHER_ACTION_SAMPLES:-1}"

CRITIC_UPDATES_PER_STEP="${CRITIC_UPDATES_PER_STEP:-2}"
CRITIC_SCHEDULER_STEP_MODE="${CRITIC_SCHEDULER_STEP_MODE:-outer_step}"
CRITIC_BACKUP_ACTION_SAMPLES="${CRITIC_BACKUP_ACTION_SAMPLES:-2}"
OOD_M_ACTIONS="${OOD_M_ACTIONS:-4}"
CRITIC_ACTION_WEIGHTS_STR="${CRITIC_ACTION_WEIGHTS_STR:-2 2 2 1}"
DIST_PENALTY_BETA="${DIST_PENALTY_BETA:-5.0}"
DIST_CLAMP_MAX="${DIST_CLAMP_MAX:-10}"
CRITIC_USE_STATE_ENCODER="${CRITIC_USE_STATE_ENCODER:-true}"
CRITIC_USE_INDEPENDENT_ENCODER="${CRITIC_USE_INDEPENDENT_ENCODER:-false}"
CRITIC_VISION_FREEZE_LAYERS="${CRITIC_VISION_FREEZE_LAYERS:-8}"
USE_VLM_BACKBONE_ENCODE="${USE_VLM_BACKBONE_ENCODE:-true}"

PHASE2_GLOBAL_SAMPLES="${PHASE2_GLOBAL_SAMPLES:-2}"
PHASE2_LOCAL_SAMPLES="${PHASE2_LOCAL_SAMPLES:-2}"
PHASE2_LOCAL_OPT_STEPS="${PHASE2_LOCAL_OPT_STEPS:-3}"
PHASE2_LOCAL_OPT_LR="${PHASE2_LOCAL_OPT_LR:-3e-4}"
PHASE2_LOCAL_GRAD_NORMALIZE="${PHASE2_LOCAL_GRAD_NORMALIZE:-true}"
PHASE2_LOCAL_ADV_WEIGHT="${PHASE2_LOCAL_ADV_WEIGHT:-false}"
PHASE2_LOCAL_ADV_EPS="${PHASE2_LOCAL_ADV_EPS:-1e-6}"
PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL="${PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL:-true}"
PHASE2_USE_TEACHER_ANCHOR="${PHASE2_USE_TEACHER_ANCHOR:-false}"
PHASE2_LOSS_RANK_WEIGHT="${PHASE2_LOSS_RANK_WEIGHT:-1}"
MASK_PADDED_ACTION_LOSS="${MASK_PADDED_ACTION_LOSS:-true}"
DISTILL_LAMBDA_W2="${DISTILL_LAMBDA_W2:-0.0}"
DISTILL_LAMBDA_OPT="${DISTILL_LAMBDA_OPT:-1.0}"

JOB_NAME="${JOB_NAME:-vgas_distill_${SHOT_LABEL,,}_${TASK_SPLIT}_$(date +%m%d_%H%M)}"
LOG_DIR="${LOG_DIR:-logs/${JOB_NAME}}"
PYTHON_PID_FILE="${PYTHON_PID_FILE:-}"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%F_%H-%M-%S).log"
  LOG_FILE_ABS="$(readlink -f "${LOG_FILE}")"
  PYTHON_PID_FILE_RUN="${LOG_DIR}/${JOB_NAME}.python.pid"
  rm -f "${PYTHON_PID_FILE_RUN}"
  nohup env JOB_NAME="${JOB_NAME}" LOG_DIR="${LOG_DIR}" PYTHON_PID_FILE="${PYTHON_PID_FILE_RUN}" \
    bash "$0" "$@" >"${LOG_FILE_ABS}" 2>&1 &
  SHELL_PID="$!"
  PYTHON_PID=""
  for _ in {1..50}; do
    if [[ -s "${PYTHON_PID_FILE_RUN}" ]]; then
      PYTHON_PID="$(cat "${PYTHON_PID_FILE_RUN}" 2>/dev/null || true)"
      break
    fi
    sleep 0.1
  done
  echo "started in background: shell_pid=${SHELL_PID} python_pid=${PYTHON_PID:-pending} output_dir=${OUTPUT_BASE}/${JOB_NAME} log= ${LOG_FILE_ABS}"
  exit 0
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy path does not exist: ${POLICY_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Dataset root must contain meta/info.json, got: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -n "${CRITIC_INIT_CHECKPOINT}" && ! -e "${CRITIC_INIT_CHECKPOINT}" ]]; then
  echo "Critic init checkpoint does not exist: ${CRITIC_INIT_CHECKPOINT}" >&2
  exit 1
fi

NEEDS_TEACHER=false
if [[ "${PHASE2_USE_TEACHER_ANCHOR}" == "true" ]]; then
  NEEDS_TEACHER=true
fi
if [[ "${PHASE1_USE_STUDENT_AS_SOURCE}" != "true" ]]; then
  NEEDS_TEACHER=true
fi
if [[ "${PHASE1_TRAIN_STUDENT}" == "true" ]]; then
  NEEDS_TEACHER=true
fi
if [[ "${NEEDS_TEACHER}" == "true" && -z "${TEACHER_POLICY_PATH}" ]]; then
  TEACHER_POLICY_PATH="${POLICY_PATH}"
fi

read -r -a CRITIC_ACTION_WEIGHTS <<< "${CRITIC_ACTION_WEIGHTS_STR}"
if [[ "${#CRITIC_ACTION_WEIGHTS[@]}" -eq 0 ]]; then
  echo "CRITIC_ACTION_WEIGHTS_STR must contain at least one weight." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"

CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate VGAS

EXTRA_FLAGS=()
if [[ "${PHASE2_INCLUDE_DATASET_ACTION_IN_GLOBAL_POOL}" == "true" ]]; then
  EXTRA_FLAGS+=(--phase2-include-dataset-action-in-global-pool)
fi
if [[ "${WANDB_ENABLE:-true}" == "true" ]]; then
  EXTRA_FLAGS+=(
    --wandb
    "--wandb-mode=${WANDB_MODE:-online}"
    "--wandb-project=${WANDB_PROJECT:-VGAS_MetaWorld_Distill}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    EXTRA_FLAGS+=("--wandb-entity=${WANDB_ENTITY}")
  fi
fi
if [[ -n "${TEACHER_POLICY_PATH}" ]]; then
  EXTRA_FLAGS+=("--teacher-policy-path=${TEACHER_POLICY_PATH}")
fi
if [[ -n "${DIST_CLAMP_MAX}" ]]; then
  EXTRA_FLAGS+=("--dist-clamp-max=${DIST_CLAMP_MAX}")
fi
if [[ -n "${PHASE2_LOSS_RANK_WEIGHT}" ]]; then
  EXTRA_FLAGS+=("--phase2-loss-rank-weight=${PHASE2_LOSS_RANK_WEIGHT}")
fi
if [[ -n "${POLICY_LR}" ]]; then
  EXTRA_FLAGS+=("--policy-lr=${POLICY_LR}")
fi
if [[ -n "${POLICY_SCHEDULER_WARMUP_STEPS}" ]]; then
  EXTRA_FLAGS+=("--policy-scheduler-warmup-steps=${POLICY_SCHEDULER_WARMUP_STEPS}")
fi
if [[ -n "${POLICY_SCHEDULER_DECAY_STEPS}" ]]; then
  EXTRA_FLAGS+=("--policy-scheduler-decay-steps=${POLICY_SCHEDULER_DECAY_STEPS}")
fi
if [[ -n "${POLICY_SCHEDULER_DECAY_LR}" ]]; then
  EXTRA_FLAGS+=("--policy-scheduler-decay-lr=${POLICY_SCHEDULER_DECAY_LR}")
fi

CRITIC_INIT_FLAGS=()
if [[ -n "${CRITIC_INIT_CHECKPOINT}" ]]; then
  CRITIC_INIT_FLAGS+=("--critic-init-checkpoint=${CRITIC_INIT_CHECKPOINT}")
fi

PHASE1_FLAGS=()
if [[ -n "${PHASE1_CRITIC_UPDATES_PER_STEP}" ]]; then
  PHASE1_FLAGS+=("--phase1-critic-updates-per-step=${PHASE1_CRITIC_UPDATES_PER_STEP}")
fi

TRAIN_CMD=(
  python scripts/run_qchunk_offline.py
  --distill
  "--policy-path=${POLICY_PATH}"
  "${CRITIC_INIT_FLAGS[@]}"
  "--output-dir=${OUTPUT_BASE}"
  --output-dir-layout=tag
  "--job-name=${JOB_NAME}"
  "--dataset-repo-id=${DATASET_REPO_ID}"
  "--dataset-root=${DATASET_ROOT}"
  --env-type=none
  --eval-freq=0
  "--steps=${STEPS}"
  "--batch-size=${BATCH_SIZE}"
  "--num-workers=${NUM_WORKERS}"
  "--log-interval=${LOG_INTERVAL}"
  "--checkpoint-interval=${CHECKPOINT_INTERVAL}"
  "--seed=${SEED}"
  "--chunk-size=${CHUNK_SIZE}"
  "--n-action-steps=${N_ACTION_STEPS}"
  "--q-chunk-len=${Q_CHUNK_LEN}"
  "--student-num-steps=${STUDENT_NUM_STEPS}"
  --critic-type=q_chunk_former
  --critic-lr=1e-4
  --critic-lr-final=2.5e-5
  "--critic-total-steps=${STEPS}"
  "--critic-scheduler-step-mode=${CRITIC_SCHEDULER_STEP_MODE}"
  "--critic-updates-per-step=${CRITIC_UPDATES_PER_STEP}"
  "--critic-backup-action-samples=${CRITIC_BACKUP_ACTION_SAMPLES}"
  --critic-q-agg=min
  --critic-loss-mode=per_head_mean
  --critic-att-mode=bi-level
  "--critic-use-state-encoder=${CRITIC_USE_STATE_ENCODER}"
  "--critic-use-independent-encoder=${CRITIC_USE_INDEPENDENT_ENCODER}"
  "--critic-vision-freeze-layers=${CRITIC_VISION_FREEZE_LAYERS}"
  "--use-vlm-backbone-encode=${USE_VLM_BACKBONE_ENCODE}"
  --use-calql=false
  --use-ood-reg=true
  "--ood-m-actions=${OOD_M_ACTIONS}"
  --ood-alpha=5.0
  "--dist-penalty-beta=${DIST_PENALTY_BETA}"
  --use-ood-noise=true
  --use-ood-trunc=false
  --use-ood-mix=true
  --ood-noise-stds 0.02
  --ood-mix-ratio=1.0
  --ood-mix-alpha-low=0.2
  --ood-mix-alpha-high=0.8
  --loss-rank-weight=5.0
  --use-raw-state-fusion=true
  --raw-state-dim=4
  --critic-grad-clip=10
  --critic-action-weights "${CRITIC_ACTION_WEIGHTS[@]}"
  --discount=0.98
  --critic-mask-dropout-prob=0.0
  --ood-action-source=erg
  "--distill-phase1-steps=${DISTILL_PHASE1_STEPS}"
  "--phase1-teacher-action-samples=${PHASE1_TEACHER_ACTION_SAMPLES}"
  "--phase1-train-student=${PHASE1_TRAIN_STUDENT}"
  "--phase1-use-student-as-source=${PHASE1_USE_STUDENT_AS_SOURCE}"
  "${PHASE1_FLAGS[@]}"
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
  "${EXTRA_FLAGS[@]}"
)

echo "[run] JOB_NAME=${JOB_NAME}"
echo "[run] OUTPUT_DIR=${OUTPUT_BASE}/${JOB_NAME}"
echo "[run] DISTILL_PHASE1_STEPS=${DISTILL_PHASE1_STEPS}"
echo "[run] POLICY_LR=${POLICY_LR:-policy preset}"
echo "[run] POLICY_SCHEDULER_WARMUP_STEPS=${POLICY_SCHEDULER_WARMUP_STEPS:-policy preset}"
echo "[run] POLICY_SCHEDULER_DECAY_STEPS=${POLICY_SCHEDULER_DECAY_STEPS:-policy preset}"
echo "[run] POLICY_SCHEDULER_DECAY_LR=${POLICY_SCHEDULER_DECAY_LR:-policy preset}"
echo "[run] PHASE1_TRAIN_STUDENT=${PHASE1_TRAIN_STUDENT}"
echo "[run] PHASE1_USE_STUDENT_AS_SOURCE=${PHASE1_USE_STUDENT_AS_SOURCE}"
echo "[run] LOSS_RANK_WEIGHT=5.0"
echo "[run] PHASE2_LOSS_RANK_WEIGHT=${PHASE2_LOSS_RANK_WEIGHT}"
echo "[run] CRITIC_INIT_CHECKPOINT=${CRITIC_INIT_CHECKPOINT:-<none>}"

if [[ -n "${PYTHON_PID_FILE}" ]]; then
  "${TRAIN_CMD[@]}" &
  PYTHON_PID="$!"
  echo "${PYTHON_PID}" > "${PYTHON_PID_FILE}"
  echo "[run] PYTHON_PID=${PYTHON_PID}"
  wait "${PYTHON_PID}"
else
  "${TRAIN_CMD[@]}"
fi
