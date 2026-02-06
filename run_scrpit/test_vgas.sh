REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=1
CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate VGAS
LIBERO_ROOT=/home/chanxu/Data/workplace/vla_exp/LIBERO
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

BASE_OUT="${REPO_ROOT}/outputs/eval"
BEST_OF_N=8        
N_ACTION=20         
EVAL_BATCH=50       
N_EPISODES="$EVAL_BATCH"  
SEEDS=(0 42 1234 124 410 2000)        
MODE_LABEL=$([ "$BEST_OF_N" -le 1 ] && echo "BC" || echo "BEST_OF_N${BEST_OF_N}")
DIR_NAME="${MODE_LABEL}_EXEC_${N_ACTION}_BATCH${EVAL_BATCH}/vgas"
# PATH TO YOUR POLICY CHECKPOINTS
POLICY_PATH="${REPO_ROOT}/to-huggingface/smolvla/5_SHOT/pretrained_model"
# PATH TO YOUR CRITIC CHECKPOINTS
CRITIC_PATH="${REPO_ROOT}/to-huggingface/goal/critic_pretrained_model/last.ckpt"

for seed in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT}/${DIR_NAME}/libero_goal/seed_${seed}_test"
  mkdir -p "$OUT_DIR"
  python scripts/eval_qc_bestofn.py --env-task=libero_goal \
                                    --policy-path="${POLICY_PATH}" \
                                    --videos-dir="${OUT_DIR}" \
                                    --seed "${seed}" \
                                    --critic-state="${CRITIC_PATH}" \
                                    --use-current-critic True \
                                    --eval-batch-size "$EVAL_BATCH" \
                                    --n-action-steps "$N_ACTION" \
                                    --eval-all-suite-tasks \
                                    --n-episodes "$N_EPISODES" \
                                    --use-best-of-n \
                                    --best-of-n "$BEST_OF_N"
  if [ -f "${OUT_DIR}/summary.json" ]; then
    python - <<'PY' "${OUT_DIR}/summary.json" "${seed}"
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
