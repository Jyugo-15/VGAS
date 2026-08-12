#!/usr/bin/env bash
# Build a merged few-shot MetaWorld dataset, annotate rewards, and optionally
# split into per-group datasets.
#
# Output layout:
#   <OUT_ROOT>/merge
#   <OUT_ROOT>/easy      (optional)
#   <OUT_ROOT>/medium    (optional)
#   <OUT_ROOT>/hard      (optional)
#   <OUT_ROOT>/very_hard (optional)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/dataset/MetaWorld/MT50_50_SHOT}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/dataset/MetaWorld/MT50_5_SHOT}"
MAPPING_JSON="${MAPPING_JSON:-$REPO_ROOT/data/metaworld/metaworld_task_groups.json}"

FEW_SHOT_PER_TASK="${FEW_SHOT_PER_TASK:-5}"
FEW_SHOT_MODE="${FEW_SHOT_MODE:-random}"
FEW_SHOT_SEED="${FEW_SHOT_SEED:-0}"
TASK_IDS="${TASK_IDS:-}"

REWARD_N_LAST="${REWARD_N_LAST:-3}"
REWARD_VALUE="${REWARD_VALUE:-1.0}"
DEFAULT_REWARD="${DEFAULT_REWARD:--1.0}"
TERMINAL_MODE="${TERMINAL_MODE:-none}"

SPLIT_GROUPS="${SPLIT_GROUPS:-false}"
DEFAULT_GROUPS="easy medium hard very_hard"
read -r -a GROUPS_ARR <<< "${GROUPS:-$DEFAULT_GROUPS}"

FORCE="${FORCE:-true}"
FORCE_FLAG=()
if [[ "$FORCE" == "true" ]]; then
  FORCE_FLAG+=(--force)
fi

TASK_ID_FLAGS=()
if [[ -n "$TASK_IDS" ]]; then
  read -r -a TASK_ID_ARR <<< "$TASK_IDS"
  TASK_ID_FLAGS+=(--task-ids "${TASK_ID_ARR[@]}")
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
"$PYTHON_BIN" data/metaworld/split_metaworld_few_shot.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUT_ROOT/merge" \
  --few-shot-per-task "$FEW_SHOT_PER_TASK" \
  --few-shot-mode "$FEW_SHOT_MODE" \
  --few-shot-seed "$FEW_SHOT_SEED" \
  "${TASK_ID_FLAGS[@]}" \
  "${FORCE_FLAG[@]}"

echo "[step2] annotate rewards on merged dataset"
"$PYTHON_BIN" data/metaworld/annotate_metaworld_rewards.py \
  --dataset-root "$OUT_ROOT/merge" \
  --n-last "$REWARD_N_LAST" \
  --reward-value "$REWARD_VALUE" \
  --default-reward "$DEFAULT_REWARD" \
  --terminal-mode "$TERMINAL_MODE" \
  --overwrite

if [[ "$SPLIT_GROUPS" == "true" ]]; then
  echo "[step3] split merged into per-group datasets at $OUT_ROOT"
  "$PYTHON_BIN" data/metaworld/split_metaworld_by_group.py \
    --dataset-root "$OUT_ROOT/merge" \
    --mapping-json "$MAPPING_JSON" \
    --output-root "$OUT_ROOT" \
    --groups "${GROUPS_ARR[@]}" \
    "${FORCE_FLAG[@]}"
else
  echo "[step3] skip group split (SPLIT_GROUPS=$SPLIT_GROUPS)"
fi

echo "[done] merged: $OUT_ROOT/merge"
