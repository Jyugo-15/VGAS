#!/usr/bin/env bash
# Evaluate a LIBERO policy under VGAS or VGAS+.
#
#   MODE=vgas       (default) base SFT policy + Q-Chunk-Former critic,
#                   inference-time Best-of-N selection.
#   MODE=vgas_plus  the distilled / refined VGAS+ policy, run directly as a
#                   plain rollout (best-of-1, no critic, no reranking).
#
# Examples:
#   bash run_scripts/libero/test_vgas.sh                     # VGAS Best-of-N
#   MODE=vgas_plus POLICY_PATH=<distilled>/pretrained_model \
#     bash run_scripts/libero/test_vgas.sh                   # VGAS+ refined policy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl
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

MODE="${MODE:-vgas}"                 # vgas | vgas_plus
ENV_TASK="${ENV_TASK:-libero_goal}"  # example suite: libero_goal
SHOT_LABEL="${SHOT_LABEL:-5_SHOT}"
N_ACTION="${N_ACTION:-20}"
EVAL_BATCH="${EVAL_BATCH:-50}"
N_EPISODES="${N_EPISODES:-${EVAL_BATCH}}"
read -r -a SEEDS <<< "${SEEDS:-0 42 1234 124 410 2000}"
BASE_OUT="${BASE_OUT:-${REPO_ROOT}/outputs/eval}"

MODE_FLAGS=()
if [ "${MODE}" = "vgas_plus" ]; then
  # VGAS+: distilled policy, plain rollout (no critic, best-of-1).
  POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/train/smolvla_distill/${SHOT_LABEL}/${ENV_TASK}/YOUR_VGAS_PLUS_RUN/checkpoints/020000/pretrained_model}"
  MODE_FLAGS+=(--no-use-best-of-n --best-of-n 1)
else
  # VGAS: base SFT policy + critic, inference-time Best-of-N.
  POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/pretrained_vla/Libero/smolvla/${SHOT_LABEL}/pretrained_model}"
  CRITIC_PATH="${CRITIC_PATH:-${REPO_ROOT}/outputs/train/smolvla/libero/${ENV_TASK}/${SHOT_LABEL}/YOUR_CRITIC_RUN/checkpoints/020000/critic_pretrained_model/last.ckpt}"
  BEST_OF_N="${BEST_OF_N:-8}"
  MODE_FLAGS+=(--critic-state="${CRITIC_PATH}" --use-current-critic=true --use-best-of-n --best-of-n "${BEST_OF_N}")
fi

DIR_NAME="${MODE}_EXEC_${N_ACTION}_BATCH${EVAL_BATCH}"

for seed in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT}/${DIR_NAME}/${ENV_TASK}/seed_${seed}"
  mkdir -p "${OUT_DIR}"
  python scripts/eval_qc_bestofn.py \
    --env-task="${ENV_TASK}" \
    --policy-path="${POLICY_PATH}" \
    --videos-dir="${OUT_DIR}" \
    --seed "${seed}" \
    --eval-batch-size "${EVAL_BATCH}" \
    --n-action-steps "${N_ACTION}" \
    --eval-all-suite-tasks \
    --n-episodes "${N_EPISODES}" \
    "${MODE_FLAGS[@]}"

  if [ -f "${OUT_DIR}/summary.json" ]; then
    python - "${OUT_DIR}/summary.json" "${seed}" <<'PY'
import json, sys
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
