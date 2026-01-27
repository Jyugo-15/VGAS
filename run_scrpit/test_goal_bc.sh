REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
JOB_NAME="test_goal"
LOG_ROOT="${LOG_ROOT:-test_log}"
CONDA_BASE="${CONDA_BASE:-${HOME}/Data/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate ript_vla

# 输出目录命名相关，可按需修改
DATE="2025-12-26"
BASE_OUT="${REPO_ROOT}/outputs/eval/${DATE}"
BEST_OF_N=1         
N_ACTION=20        
# Q_AGG="mean"      
EVAL_BATCH=50       
N_EPISODES="$EVAL_BATCH" 
SEEDS=(0 42 1234 124 410 2000)        
TASK_ID=""          

MODE_LABEL=$([ "$BEST_OF_N" -le 1 ] && echo "BC" || echo "BEST_OF_N${BEST_OF_N}")
DIR_NAME="${MODE_LABEL}_EXEC_${N_ACTION}_BATCH${EVAL_BATCH}"
POLICY_PATH="${REPO_ROOT}/pretrained_vla/smolvla/5-shot/pretrained_model"
JOB_NAME="${JOB_NAME}_${MODE_LABEL}"

BON_FLAGS=()
if [ "$BEST_OF_N" -le 1 ]; then
  BON_FLAGS+=(--no-use-best-of-n --best-of-n 1)
else
  BON_FLAGS+=(--use-best-of-n --best-of-n "$BEST_OF_N")
fi
QAGG_FLAGS=()
[ -n "$Q_AGG" ] && QAGG_FLAGS+=(--critic-q-agg-eval "$Q_AGG")
COMMON_FLAGS=(--eval-batch-size "$EVAL_BATCH" --n-action-steps "$N_ACTION" --n-episodes "$N_EPISODES")
COMMON_FLAGS+=("${BON_FLAGS[@]}")
COMMON_FLAGS+=("${QAGG_FLAGS[@]}")
TASK_FLAGS=()
if [ -n "$TASK_ID" ]; then
  TASK_FLAGS+=(--env-task-ids "$TASK_ID")
else
  TASK_FLAGS+=(--eval-all-suite-tasks)
fi

# 外层 critic，内层 step（按需改 tag 和路径）
CRITIC_CONFIGS=(

  "goal_test|${REPO_ROOT}/outputs/train/12.27/goal_vgas/checkpoints"
)

CHECKPOINT_STEPS=(18000)
for seed in "${SEEDS[@]}"; do
  for cfg in "${CRITIC_CONFIGS[@]}"; do
    IFS='|' read -r CRITIC_TAG CRITIC_DIR <<<"$cfg"
    CRITIC_OUT_ROOT="${BASE_OUT}/${DIR_NAME}/${CRITIC_TAG}/libero_goal/seed_${seed}"
    mkdir -p "$CRITIC_OUT_ROOT"
    for step in "${CHECKPOINT_STEPS[@]}"; do
    step_pad=$(printf "%06d" "$step")
    OUT_GO="${CRITIC_OUT_ROOT}/${step_pad}"
    SUMMARY_PATH="${OUT_GO}/summary.json"

    if [ -f "$SUMMARY_PATH" ]; then
      echo "[skip] found ${SUMMARY_PATH}"
      continue
    fi

    CRITIC_FLAGS=()
    if [ "$BEST_OF_N" -gt 1 ]; then
      CRITIC_PATH="${CRITIC_DIR}/${step_pad}/critic_pretrained_model/last.ckpt"
      if [ ! -f "$CRITIC_PATH" ]; then
        echo "[skip] missing ${CRITIC_PATH}"
        continue
      fi
      CRITIC_FLAGS+=(--critic-state="${CRITIC_PATH}" --use-current-critic True)
    fi

      mkdir -p "$OUT_GO"
      LOG_DIR="${LOG_ROOT}/${JOB_NAME}/${CRITIC_TAG}/step_${step_pad}/seed_${seed}"
      mkdir -p "$LOG_DIR"
      LOG_FILE="${LOG_DIR}/eval.log"
      echo "[run] log= ${LOG_FILE}"
      python scripts/eval_qc_bestofn.py --env-task=libero_goal \
                                        --policy-path="${POLICY_PATH}" \
                                        --videos-dir="${OUT_GO}" \
                                        "${CRITIC_FLAGS[@]}" \
                                        --seed "${seed}" \
                                        "${COMMON_FLAGS[@]}" \
                                        "${TASK_FLAGS[@]}" \
                                        >"$LOG_FILE" 2>&1 &
      PY_PID=$!
      echo "[pid] ${PY_PID}"
      status=0
      wait "${PY_PID}" || status=$?
      if [ "$status" -ne 0 ]; then
        echo "[error] python exited with status ${status} (log=${LOG_FILE})"
        tail -n 50 "$LOG_FILE"
      fi
      # 将 seed 记录进 summary.json（如已生成）
      if [ -f "${OUT_GO}/summary.json" ]; then
        python - <<'PY' "${OUT_GO}/summary.json" "${seed}"
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
  done
done
