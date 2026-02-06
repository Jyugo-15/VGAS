#!/usr/bin/env bash
# Build a merged 5-shot dataset, annotate rewards, then split into per-suite datasets.
# Output layout:
#   <OUT_ROOT>/merge
#   <OUT_ROOT>/libero_object
#   <OUT_ROOT>/libero_goal
#   <OUT_ROOT>/libero_spatial
#   <OUT_ROOT>/libero_10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/dataset/Libero/HFlibero}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/dataset/Libero/HF_LIBERO_5_SHOT_TEST}"
MAPPING_JSON="${MAPPING_JSON:-$REPO_ROOT/data/libero_tasksuit_info/libero_task_suites.json}"

FEW_SHOT_PER_TASK="${FEW_SHOT_PER_TASK:-5}"
FEW_SHOT_MODE="${FEW_SHOT_MODE:-random}"
FEW_SHOT_SEED="${FEW_SHOT_SEED:-0}"

DEFAULT_SUITES="libero_object libero_goal libero_spatial libero_10"
read -r -a SUITES_ARR <<< "${SUITES:-$DEFAULT_SUITES}"

REWARD_N_LAST="${REWARD_N_LAST:-3}"
REWARD_VALUE="${REWARD_VALUE:-1.0}"
DEFAULT_REWARD="${DEFAULT_REWARD:-0.0}"
REWARD_DISCOUNT="${REWARD_DISCOUNT:-0.98}"

FORCE="${FORCE:-true}"
FORCE_FLAG=()
if [[ "$FORCE" == "true" ]]; then
  FORCE_FLAG+=(--force)
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[error] DATASET_ROOT not found: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "[error] MAPPING_JSON not found: $MAPPING_JSON" >&2
  exit 1
fi

echo "[step1] build merged few-shot at ${OUT_ROOT}/merge"
"$PYTHON_BIN" data/split_hflibero_few_shot.py \
  --dataset-root "$DATASET_ROOT" \
  --mapping-json "$MAPPING_JSON" \
  --output-root "$OUT_ROOT/merge" \
  --suites "${SUITES_ARR[@]}" \
  --few-shot-per-task "$FEW_SHOT_PER_TASK" \
  --few-shot-mode "$FEW_SHOT_MODE" \
  --few-shot-seed "$FEW_SHOT_SEED" \
  "${FORCE_FLAG[@]}"

echo "[step2] annotate rewards on merged dataset"
"$PYTHON_BIN" data/annotate_rewards.py \
  --dataset-root "$OUT_ROOT/merge" \
  --n-last "$REWARD_N_LAST" \
  --reward-value "$REWARD_VALUE" \
  --default-reward "$DEFAULT_REWARD" \
  --discount "$REWARD_DISCOUNT" \
  --overwrite

echo "[step3] split merged into per-suite datasets at $OUT_ROOT"
"$PYTHON_BIN" data/split_hflibero_by_suite.py \
  --dataset-root "$OUT_ROOT/merge" \
  --mapping-json "$MAPPING_JSON" \
  --output-root "$OUT_ROOT" \
  --suites "${SUITES_ARR[@]}" \
  "${FORCE_FLAG[@]}"

echo "[done] merged: $OUT_ROOT/merge | per-suite: $OUT_ROOT/<suite>"
