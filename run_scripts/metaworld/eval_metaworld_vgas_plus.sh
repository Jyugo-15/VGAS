#!/usr/bin/env bash
# Evaluate the VGAS+ (distilled / refined) MetaWorld policy.
#
# VGAS+ produces a refined policy that generates high-value action chunks
# directly, so it is evaluated WITHOUT the critic and WITHOUT Best-of-N
# reranking -- i.e. a plain policy rollout, identical to the BC evaluation but
# pointed at the distilled checkpoint produced by
# run_scripts/metaworld/train_vgas_distill_metaworld.sh.
#
# This is a thin wrapper that sets the distilled-checkpoint location and
# delegates to eval_metaworld_bc.sh (the shared plain-eval engine).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Distilled (VGAS+) checkpoints dir: a "<run>/checkpoints" folder that contains
# "<step>/pretrained_model". Point this at the output of
# train_vgas_distill_metaworld.sh (TASK_SPLIT defaults to very_hard).
export CKPT_ROOT_REL="${CKPT_ROOT_REL:-outputs/train/smolvla_distill/metaworld/5_SHOT/very_hard/YOUR_VGAS_PLUS_RUN/checkpoints}"
export STEPS_STR="${STEPS_STR:-30000}"
export TASK_GROUPS_STR="${TASK_GROUPS_STR:-very_hard}"  # example: very_hard
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output_metaworld/VGAS_PLUS_5_SHOT}"
export JOB_NAME="${JOB_NAME:-eval_metaworld_vgas_plus}"

exec "${SCRIPT_DIR}/eval_metaworld_bc.sh" "$@"
