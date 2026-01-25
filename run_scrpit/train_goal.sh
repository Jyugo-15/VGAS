# 在原版本temp_script/12-24/train_5000_goal_final.sh 上修改training step, lr,以及 finial lr， 其他的暂时不变
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_NAME="goal_dis_5_alpha_5_pair_t_trunc_f_att_f_mix_1_edit_step_le_finial_lr"
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

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
CONDA_BASE="/home/chanxu/Data/anaconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate ript_vla
python scripts/train_qchunk_offline.py \
  --policy.path="${REPO_ROOT}/pretrained_vla/smolvla/5-shot/pretrained_model" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=20 \
  --dataset.repo_id=libero_goal \
  --dataset.root="${REPO_ROOT}/dataset/Libero/HF_LIBERO_5SHORT/libero_goal" \
  --job_name="${JOB_NAME}" \
  --steps=20000 \
  --log_freq=50 \
  --batch_size=32 \
  --critic_only=true \
  --q_chunk_len=32 \
  --critic.q_chunk_len=32 \
  --critic.lr=1e-4 \
  --critic.lr_final=2.5e-5 \
  --critic.lr_total_steps=20000 \
  --critic.q_aggregation=min \
  --critic.critic_loss_mode=per_head_mean \
  --critic.att_mode=bi-level \
  --critic.use_calql=false \
  --critic.use_ood_reg=true \
  --critic.dist_penalty_beta=5.0 \
  --critic.dist_clamp_max=20.0 \
  --critic.use_raw_state_fusion=true \
  --critic.grad_clip_norm=10 \
  --critic.action_distance_weights='[5,5,5,1,1,1,1]' \
  --critic.discount=0.98 \
  --critic.mask_dropout_prob=0.0 \
  --critic.ood_alpha=5.0 \
  --critic.ood_action_source=erg \
  --critic.use_ood_noise=true \
  --critic.use_ood_trunc=false \
  --critic.use_ood_mix=true \
  --critic.debug_mix_dist=false \
  --critic.ood_noise_stds='[0.02]' \
  --critic.ood_mix_ratio=1.0 \
  --critic.ood_mix_alpha_low=0.2 \
  --critic.ood_mix_alpha_high=0.8 \
  --critic.use_pairwise_ood_loss=true \
  --critic.pairwise_ood_loss_weight=1.0 \
  --critic.eval_ranking_freq=500 \
  --critic.eval_ranking_batches=128 \
  --critic.eval_ranking_action_samples=8 \
  --critic.eval_ranking_batch_size=32 \
  --critic.eval_ranking_full_dataset_root="${DATASET_ROOT}" \
  --wandb.enable=true \
  --wandb.mode=online
