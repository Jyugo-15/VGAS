
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

JOB_NAME="train_goal"
LOG_DIR="${LOG_DIR:-logs/${JOB_NAME}}"
if [[ "${1:-}" == "--nohup" ]]; then
  shift
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%F_%H-%M-%S).log"
  LOG_FILE_ABS="$(readlink -f "$LOG_FILE")"
  nohup bash "$0" "$@" >"$LOG_FILE_ABS" 2>&1 &
  echo "started in background: pid=$! log=$LOG_FILE_ABS"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
POLICY_PATH="${REPO_ROOT}/pretrained_vla/smolvla/5-shot/pretrained_model"
DATASET_ROOT="${REPO_ROOT}/dataset/Libero/HF_LIBERO_5SHORT/libero_goal"
EVAL_DATASET_ROOT="${REPO_ROOT}/dataset/Libero/HF_LIBERO_split/libero_goal"
CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate ript_vla
python scripts/run_qchunk_offline.py \
  --policy-path="${POLICY_PATH}" \
  --chunk-size=32 \
  --ood-m-actions=4 \
  --n-action-steps=20 \
  --dataset-repo-id=libero_goal \
  --dataset-root="${DATASET_ROOT}" \
  --job-name="${JOB_NAME}" \
  --critic-type=q_chunk_former \
  --steps=20000 \
  --log-interval=50 \
  --batch-size=32 \
  --critic-only \
  --q-chunk-len=32 \
  --critic-lr=1e-4 \
  --critic-lr-final=2.5e-5 \
  --critic-total-steps=20000 \
  --critic-q-agg=min \
  --critic-loss-mode=per_head_mean \
  --critic-att-mode=bi-level \
  --use-calql=false \
  --use-ood-reg=true \
  --dist-penalty-beta=5.0 \
  --dist-clamp-max=10.0 \
  --use-raw-state-fusion=true \
  --critic-grad-clip=10 \
  --critic-action-weights 5 5 5 1 1 1 1 \
  --critic-discount=0.98 \
  --critic-mask-dropout-prob=0.0 \
  --ood-alpha=5.0 \
  --ood-action-source=erg \
  --use-ood-noise=true \
  --use-ood-trunc=false \
  --use-ood-mix=true \
  --debug-mix-dist=false \
  --ood-noise-stds 0.02 \
  --ood-mix-ratio=1.0 \
  --ood-mix-alpha-low=0.2 \
  --ood-mix-alpha-high=0.8 \
  --loss-rank-weight=5.0 \
  --eval-ranking-freq=500 \
  --eval-ranking-batches=128 \
  --eval-ranking-action-samples=8 \
  --eval-ranking-batch-size=32 \
  --checkpoint-interval 2000 \
  --eval-ranking-full-dataset-root="${EVAL_DATASET_ROOT}" \
  --wandb \
  --wandb-mode=online
