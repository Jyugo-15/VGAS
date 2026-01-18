"""Train SmolVLA with the critic-augmented QC pipeline."""

import argparse
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from pprint import pformat
import sys
from typing import Optional

# official ckpt /home/chanxu/Data/workplace/vla_exp/lerobot/models/smolvla_libero
# my train policy /home/chanxu/Data/workplace/vla_exp/lerobot/outputs/train/my_smolvla_20251010_182053/checkpoints/011000/pretrained_model
# pretrain policy /home/chanxu/Data/workplace/vla_exp/lerobot/models/smolvla_base_fix
# merge 5 shot /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/offline_qc_target_discount_few_shot_full_data_20251117_233140/checkpoints/010000/pretrained_model


DEFAULT_POLICY_PATH = Path(
    "/home/chanxu/Data/workplace/vla_exp/lerobot/outputs/train/my_smolvla_new_5_shot/checkpoints/005000/pretrained_model"
)

# 全数据集add reward /home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero_AR
# 全数据集 fewshot add reward  /data/chanxu/workplace/vla_exp/lerobot/dataset/HFlibero_with_rewards/merged
DEFAULT_DATASET_ROOT = Path("/home/chanxu/Data/workplace/vla_exp/lerobot/dataset_news/HF_LIBERO_5SHORT/libero_object")
DEFAULT_DATASET_REPO_ID = "libero_object"
DEFAULT_JOB_NAME: Optional[str] = None
DEFAULT_WANDB_PROJECT = "my_project"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from utils import init_logging

try:
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import FeatureType
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import env_to_policy_features

    from scripts.train_qchunk_offline import (
        CriticConfig,
        TrainWithCriticPipelineConfig,
        train_from_config as lerobot_train,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import LeRobot dependencies. Ensure lerobot is installed or cloned alongside this repository. "
        f"Original error: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    def str2bool(value: str) -> bool:
        value = value.lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError("Expected a boolean value.")

    parser = argparse.ArgumentParser(description="Train SmolVLA with QC critic augmentation.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH, help="Pretrained checkpoint to finetune.")
    parser.add_argument("--policy-config", type=Path, default= None, help="Optional SmolVLA config.json to load (defaults to <policy-path>/config.json).") #"/home/chanxu/Data/workplace/vla_exp/lerobot/models/smolvla_libero/config.json"
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Local dataset root if using local storage.")
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID, help="LeRobot dataset repo identifier.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Optional subset of episode indices.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"), help="Base directory for outputs.")
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="Custom run name (defaults to q_agg_* auto naming when omitted).")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--config-path", type=Path, default=None, help=f"Path to an existing `{TRAIN_CONFIG_NAME}` when using `--resume`.")
    parser.add_argument("--steps", type=int, default=12000, help="Number of optimisation steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for policy and data.")
    parser.add_argument("--chunk-size", type=int, default=32, help="Action chunk size for SmolVLA.")
    parser.add_argument("--n-action-steps", type=int, default=20, help="Number of supervised action steps.")
    parser.add_argument("--q-chunk-len", type=int, default=32, help="Critic/future-observation chunk length (defaults to --n-action-steps).")
    parser.add_argument("--discount", type=float, default=None, help="Discount factor (used for critic and mc returns).")
    parser.add_argument("--obs-steps", type=int, default=1, help="Number of observation steps provided to the model.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging frequency in steps.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint frequency in steps.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluation frequency during training (0 disables).")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for training components.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming dataset loading mode.")
    parser.add_argument("--expert-width-multiplier", type=float, default=None, help="Width multiplier for the LM/expert layers (must match checkpoint).")
    parser.add_argument("--disable-save-checkpoint", action="store_true", help="Disable checkpoint saving even if LeRobot would normally save.")
    
    parser.add_argument("--use-data-augmentations",default=False, type=bool, help="Enable or disable visual data augmentations in encode_policy_observations_test.")
    parser.add_argument("--use-vlm-backbone-encode",default=True,help="Pass embeddings through VLM backbone encode in encode_policy_observations_test.",)
    
    parser.add_argument("--load-vlm-weights", default=True, type=bool, help="Load the VLM backbone weights.")
    parser.add_argument("--unfreeze-vision-encoder", action="store_true", help="Finetune the vision encoder layers.")
    parser.add_argument("--train-full-model", action="store_true", help="Disable expert-only training.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push checkpoints to the Hugging Face Hub.")
    parser.add_argument("--policy-repo-id", type=str, default=None, help="Hub repo id when pushing to the hub.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--critic-only", dest="critic_only", action="store_true", help="Only train the critic (skip policy updates).")
    parser.set_defaults(critic_only=True)
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/organization.")
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="W&B logging mode override.")
    parser.add_argument("--wandb-notes", type=str, default=None, help="Optional notes attached to W&B run.")
    parser.add_argument("--wandb-upload-policy", dest="wandb_upload_policy", action="store_true", help="Upload policy checkpoints to W&B artifacts.")
    parser.add_argument("--no-wandb-upload-policy", dest="wandb_upload_policy", action="store_false", help="Disable uploading policy checkpoints to W&B.")
    parser.add_argument("--wandb-upload-code", dest="wandb_upload_code", action="store_true", help="Upload repository source code to W&B once at run start.")
    parser.add_argument("--no-wandb-upload-code", dest="wandb_upload_code", action="store_false", help="Disable uploading source code artifacts to W&B.")
    parser.set_defaults(wandb_upload_policy=False, wandb_upload_code=False)
    parser.add_argument("--use-calql", type=str2bool, default=False, help="Enable CalQL regularization in critic (True/False).")
    parser.add_argument("--use-ood-reg", type=str2bool, default=True, help="Enable explicit OOD penalty regularization.")
    parser.add_argument("--ood-alpha", type=float, default=2.0, help="Weight for OOD regularization term.")
    parser.add_argument("--dist-penalty-beta", type=float, default=5, help="Slope for distance-based OOD target.")
    parser.add_argument("--ood-warmup-steps", type=int, default=0, help="Warmup steps before enabling OOD regularization.")
    parser.add_argument(
        "--ood-include-current-actions",
        type=str2bool,
        default=True,
        help="Include policy current actions when building OOD samples.",
    )
    parser.add_argument(
        "--ood-include-random-actions",
        type=str2bool,
        default=False,
        help="Include random/noise actions when building OOD samples.",
    )
    parser.add_argument(
        "--ood-include-next-actions",
        type=str2bool,
        default=False,
        help="Include next-state actions when building OOD samples (CalQL forces True).",
    )

    # Environment helpers.
    parser.add_argument("--env-type", type=str, default="libero", help="Environment type to attach (use 'none' to disable, e.g. 'libero').")
    parser.add_argument("--env-task", type=str, default="libero_object", help="Libero task identifier when env-type=libero.")
    parser.add_argument("--env-obs-type", type=str, default="pixels_agent_pos", help="Observation type for Libero env (pixels or pixels_agent_pos).")
    parser.add_argument("--env-camera-name", type=str, default="agentview_image,robot0_eye_in_hand_image", help="Comma-separated camera names for Libero env.")
    parser.add_argument("--env-fps", type=int, default=30, help="Frame rate for Libero env.")
    parser.add_argument("--env-episode-length", type=int, default=520, help="Episode length for Libero env.")
    parser.add_argument("--env-disable-init-states", action="store_true", help="Disable loading stored initial states for the Libero environment.")

    # Critic-specific knobs (mirrors CriticConfig defaults).
    parser.add_argument("--critic-disable", default=False, help="Disable critic updates (pure BC).")
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512], help="Hidden dimensions for the critic backbone MLP.")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Critic optimizer learning rate.")
    parser.add_argument("--critic-betas", type=float, nargs=2, default=[0.9, 0.95], metavar=("BETA1", "BETA2"), help="Critic Adam betas.")
    parser.add_argument("--critic-weight-decay", type=float, default=1e-10, help="Weight decay applied to critic parameters.")
    parser.add_argument("--critic-discount", type=float, default=0.98, help="Discount factor for critic targets.")
    parser.add_argument("--critic-tau", type=float, default=0.005, help="Soft target update coefficient.")
    parser.add_argument(
        "--critic-tau-warmup",
        type=float,
        default=None,
        help="Optional tau used during warmup steps before switching to --critic-tau.",
    )
    parser.add_argument(
        "--critic-tau-warmup-steps",
        type=int,
        default=0,
        help="Number of initial steps to use warmup tau (if provided).",
    )
    parser.add_argument("--critic-grad-clip", type=float, default=10.0, help="Gradient clipping norm for critic updates.")
    parser.add_argument(
        "--critic-grad-clip-warmup",
        type=float,
        default=None,
        help="Optional gradient clip to use during warmup steps (uses --ood-warmup-steps as boundary).",
    )
    parser.add_argument("--critic-action-samples", type=int, default=8, help="Number of candidate chunks evaluated in the best-of-n sampler.")
    parser.add_argument("--best-of-n-samples", type=int, default=None, help="Override for the number of Best-of-N samples (defaults to --critic-action-samples).")
    parser.add_argument("--critic-q-agg", type=str, default="min", choices=["mean", "min", "max"], help="Aggregation used when combining twin Q estimates.")
    parser.add_argument("--critic-att-mode", type=str, default="bi-level", choices=["causal", "bi-level"], help="Attention pattern for transformer-based critic heads.")
    parser.add_argument("--critic-temperature", type=float, default=2.0, help="Noise scale for candidate actions.")
    parser.add_argument(
        "--critic-mask-dropout-prob",
        type=float,
        default=0.5,
        help="Dropout prob for action padding mask during critic training (0 disables).",
    )
    parser.add_argument(
        "--critic-value-head-bias-enable",
        action="store_true",
        help="Initialize value head final-layer bias to a constant.",
    )
    parser.add_argument(
        "--critic-value-head-bias-value",
        type=float,
        default=0.0,
        help="Constant bias value when --critic-value-head-bias-enable is set.",
    )
    parser.add_argument(
        "--critic-use-dual-noise-ood",
        action="store_true",
        help="Use dual-noise GT negatives (tiny + small noise) instead of truncation OOD.",
    )
    parser.add_argument(
        "--critic-action-weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-dimension weights for action distance in OOD penalty (length must match action dim).",
    )
    parser.add_argument("--cql-m-actions", type=int, default=2, help="Number of CQL samples (defaults to all best-of-n candidates).")
    parser.add_argument("--cql-alpha", type=float, default=1.0, help="Weight for CalQL/CQL regularization term.")
    parser.add_argument("--cql-next-noise-std", type=float, default=0.05, help="Std for CalQL next-action noise (set 0 to disable).")
    parser.add_argument(
        "--cql-cur-noise-std",
        type=float,
        default=None,
        help="Std for CalQL current-action noise (defaults to next noise std; set 0 to disable).",
    )

    parser.add_argument("--critic-type", type=str, default="my_value_query_head", choices=["mlp", "value_query_head", "value_head","my_value_query_head","my_value_head"], help="Critic architecture to use for QC training.")
    parser.add_argument("--critic-head-type", type=str, default="transformer", choices=["mlp", "transformer"], help="Head to pair with value_query_head critics.")
    parser.add_argument("--critic-head-num-layers", type=int, default=2, help="Number of transformer layers for the critic head when applicable.")
    parser.add_argument("--critic-head-mlp-dims", type=int, nargs="+", default=None, help="MLP dimensions used inside the critic head (defaults to critic hidden dims).")
    parser.add_argument("--critic-vqh-hidden-dims", type=int, nargs="+", default=None, help="Hidden dims for ValueQueryHead backbones (defaults to critic hidden dims).")
    parser.add_argument("--critic-vqh-num-backbone-layers", type=int, default=2, help="Number of decoder layers for ValueQueryHead backbones.")
    parser.add_argument("--critic-vqh-vlm-model-name", type=str, default=None, help="Override the VLM model name used when instantiating ValueQueryHead backbones.")
    parser.add_argument("--critic-num-q-heads", type=int, default=2, help="Number of independent Q heads when using ValueQueryHead critics.")
    parser.add_argument("--critic-loss-mode", type=str, default="per_head_mean", choices=["mse", "per_head_mean"], help="Reduction applied to critic TD errors.") # mse 对各个头求平均再做mse, per_head_mean 对 各个头先计算loss再做平均
    parser.add_argument("--num-query-token", type=int, default=16, help="Number of query tokens for transformer critic heads.")
    parser.add_argument("--critic-warmup-steps", type=int, default=1000, help="Warmup steps for the critic learning rate.")
    parser.add_argument("--critic-total-steps", type=int, default=12000, help="Optional total steps for critic LR scheduling (defaults to --steps).")
    parser.add_argument("--critic-lr-final", type=float, default=2.5e-06, help="Final critic learning rate after decay (default 0, i.e. decay to zero).")
    parser.add_argument("--critic-value-head-num-layers", type=int, default=2, help="Number of transformer layers for value_head critics.")
    parser.add_argument("--critic-value-head-mlp-dims", type=int, nargs="+", default=None, help="MLP hidden dims for value_head critics (defaults to critic hidden dims).")
    parser.add_argument("--critic-value-head-vlm-model-name", type=str, default=None, help="Override VLM model when constructing value_head critics.")
    parser.add_argument(
        "--use-query-head",
        type=str2bool,
        dest="critic_query_head",
        default=False,
        help="Whether to use the query-based critic head (set False to use the no-query variant).",
    )
    parser.add_argument(
        "--use-raw-state-fusion",
        type=str2bool,
        default=True,
        help="Enable raw state fusion into critic action embeddings (requires observation.state).",
    )
    parser.add_argument(
        "--raw-state-dim",
        type=int,
        default=8,
        help="Dimension of observation.state when raw state fusion is enabled.",
    )

    return parser.parse_args()


def build_job_name(args: argparse.Namespace) -> str:
    # Allow alias: my_value_head uses my_value_query_head without query token
    if args.critic_type == "my_value_head":
        args.critic_type = "my_value_query_head"
        args.critic_query_head = False
    if args.job_name:
        return args.job_name
    layer_count = args.critic_value_head_num_layers if args.critic_type == "my_value_query_head" else args.critic_head_num_layers
    base = f"q_agg_{args.critic_q_agg}_layer_{layer_count}_crit_lm_{args.critic_loss_mode}"
    aug_tag = "aug" if getattr(args, "use_data_augmentations", True) else "noaug"
    calql_tag = "calql" if getattr(args, "use_calql", False) else "nocalql"
    return f"{base}_{aug_tag}_{calql_tag}"


def build_env_config(args: argparse.Namespace):
    if args.env_type is None:
        return None

    env_type = args.env_type.lower()
    if env_type in {"", "none", "null"}:
        return None
    if env_type == "libero":
        kwargs = {
            "task": args.env_task,
            "obs_type": args.env_obs_type,
            "camera_name": args.env_camera_name,
            "fps": args.env_fps,
            "episode_length": args.env_episode_length,
        }
        if args.env_disable_init_states:
            kwargs["init_states"] = False
        return LiberoEnv(**kwargs)

    raise ValueError(f"Unsupported environment type: {args.env_type}")


def build_critic_config(args: argparse.Namespace, q_chunk_len: int | None = None) -> CriticConfig:
    if args.critic_type == "my_value_head":
        args.critic_type = "my_value_query_head"
        args.critic_query_head = False
    head_mlp_dims = tuple(args.critic_head_mlp_dims) if args.critic_head_mlp_dims else tuple(args.critic_hidden_dims)
    vqh_hidden_dims = (
        tuple(args.critic_vqh_hidden_dims)
        if args.critic_vqh_hidden_dims
        else tuple(args.critic_hidden_dims)
    )
    value_head_mlp_dims = (
        tuple(args.critic_value_head_mlp_dims)
        if args.critic_value_head_mlp_dims
        else head_mlp_dims
    )
    return CriticConfig(
        enable=not args.critic_disable,
        hidden_dims=tuple(args.critic_hidden_dims),
        lr=args.critic_lr,
        betas=(args.critic_betas[0], args.critic_betas[1]),
        weight_decay=args.critic_weight_decay,
        discount=args.critic_discount,
        tau=args.critic_tau,
        tau_warmup=args.critic_tau_warmup,
        tau_warmup_steps=args.critic_tau_warmup_steps,
        grad_clip_norm=args.critic_grad_clip,
        grad_clip_warmup=args.critic_grad_clip_warmup,
        action_samples=args.best_of_n_samples or args.critic_action_samples,
        q_aggregation=args.critic_q_agg,
        temperature=args.critic_temperature,
        critic_type=args.critic_type,
        vqh_num_backbone_layers=args.critic_vqh_num_backbone_layers,
        vqh_hidden_dims=vqh_hidden_dims,
        vqh_vlm_model_name=args.critic_vqh_vlm_model_name,
        head_type=args.critic_head_type,
        head_num_layers=args.critic_head_num_layers,
        head_mlp_dims=head_mlp_dims,
        num_q_heads=args.critic_num_q_heads,
        critic_loss_mode=args.critic_loss_mode,
        att_mode=args.critic_att_mode,
        num_query_token=args.num_query_token,
        value_head_num_layers=args.critic_value_head_num_layers,
        value_head_mlp_dims=value_head_mlp_dims,
        value_head_vlm_model_name=args.critic_value_head_vlm_model_name,
        lr_warmup_steps=args.critic_warmup_steps,
        lr_total_steps=args.critic_total_steps or args.steps,
        lr_final=args.critic_lr_final,
        use_calql=args.use_calql,
        cql_m_actions=args.cql_m_actions,
        cql_alpha=args.cql_alpha,
        cql_next_noise_std=args.cql_next_noise_std,
        cql_cur_noise_std=(
            args.cql_cur_noise_std if args.cql_cur_noise_std is not None else args.cql_next_noise_std
        ),
        use_no_query_head=not args.critic_query_head,
        use_ood_reg=args.use_ood_reg,
        ood_alpha=args.ood_alpha,
        dist_penalty_beta=args.dist_penalty_beta,
        ood_warmup_steps=args.ood_warmup_steps,
        ood_include_current_actions=args.ood_include_current_actions,
        ood_include_random_actions=args.ood_include_random_actions,
        ood_include_next_actions=args.ood_include_next_actions,
        use_raw_state_fusion=args.use_raw_state_fusion,
        raw_state_dim=args.raw_state_dim,
        q_chunk_len=q_chunk_len,
        action_distance_weights=tuple(args.critic_action_weights) if args.critic_action_weights else None,
        mask_dropout_prob=args.critic_mask_dropout_prob,
        value_head_bias_init_enabled=args.critic_value_head_bias_enable,
        value_head_bias_init_value=args.critic_value_head_bias_value,
        use_dual_noise_ood=args.critic_use_dual_noise_ood,
    )


def build_train_config(args: argparse.Namespace) -> TrainWithCriticPipelineConfig:
    job_name = build_job_name(args)
    job_name += "test"
    date_prefix = datetime.now().strftime("%m.%d")
    output_dir = args.output_dir / date_prefix / job_name

    n_action_steps = args.n_action_steps or args.chunk_size
    if n_action_steps > args.chunk_size:
        raise ValueError(f"`n_action_steps` ({n_action_steps}) cannot exceed `chunk_size` ({args.chunk_size}).")
    q_chunk_len = args.q_chunk_len or n_action_steps
    if q_chunk_len > args.chunk_size:
        raise ValueError(f"`q_chunk_len` ({q_chunk_len}) cannot exceed `chunk_size` ({args.chunk_size}).")
    if q_chunk_len != args.chunk_size:
        raise ValueError(
            f"`q_chunk_len` ({q_chunk_len}) must match `chunk_size` ({args.chunk_size}) to keep critic time spans aligned."
        )

    policy_config_path = args.policy_config
    if policy_config_path is None and args.policy_path:
        candidate = Path(args.policy_path) / "config.json"
        if candidate.exists():
            policy_config_path = candidate
    policy_cfg_payload: dict = {}
    if policy_config_path is not None and policy_config_path.exists():
        with policy_config_path.open("r", encoding="utf-8") as f:
            policy_cfg_payload = json.load(f)
        policy_cfg_payload.pop("type", None)
    policy_cfg_payload.update(
        {
            "device": args.device,
            "pretrained_path": str(args.policy_path) if args.policy_path else None,
            "chunk_size": args.chunk_size,
            "n_action_steps": n_action_steps,
            "n_obs_steps": args.obs_steps,
            "use_amp": args.use_amp,
            "push_to_hub": args.push_to_hub,
            "repo_id": args.policy_repo_id,
            # "load_vlm_weights": args.load_vlm_weights,
            # "freeze_vision_encoder": not args.unfreeze_vision_encoder,
            # "train_expert_only": not args.train_full_model,
        }
    )
    if args.expert_width_multiplier is not None:
        policy_cfg_payload["expert_width_multiplier"] = args.expert_width_multiplier
    policy_cfg = SmolVLAConfig(**policy_cfg_payload)

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root) if args.dataset_root else None,
        episodes=args.episodes,
        streaming=args.streaming,
    )

    wandb_cfg = WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        notes=args.wandb_notes,
    )

    env_cfg = build_env_config(args)
    if env_cfg is not None:
        env_features = env_to_policy_features(env_cfg)
        policy_cfg.output_features = {
            key: feature for key, feature in env_features.items() if feature.type is FeatureType.ACTION
        }
        policy_cfg.input_features = {
            key: feature for key, feature in env_features.items() if key not in policy_cfg.output_features
        }

    critic_cfg = build_critic_config(args, q_chunk_len=q_chunk_len)
    # 对齐 critic 与 dataset 的折扣因子
    if args.discount is not None:
        critic_cfg.discount = args.discount
    if args.use_calql:
        critic_cfg.use_calql = True
    if args.use_ood_reg:
        critic_cfg.use_ood_reg = True

    train_cfg = TrainWithCriticPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        env=env_cfg,
        output_dir=output_dir,
        job_name=job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_freq=args.eval_freq,
        log_freq=args.log_interval,
        save_freq=args.checkpoint_interval,
        seed=args.seed,
        save_checkpoint=not args.disable_save_checkpoint,
        wandb=wandb_cfg,
        critic=critic_cfg,
        log_policy_to_wandb=args.wandb_upload_policy,
        log_code_to_wandb=args.wandb_upload_code,
        q_chunk_len=q_chunk_len,
    )
    # 将折扣传递给自定义数据集，保证一致
    if args.discount is not None:
        train_cfg.dataset.discount = args.discount
    if args.use_calql:
        train_cfg.critic.use_calql = True
    # Toggles forwarded to encode_policy_observations_test
    train_cfg.use_data_augmentations = args.use_data_augmentations
    train_cfg.use_vlm_backbone_encode = args.use_vlm_backbone_encode
    train_cfg.critic_only = args.critic_only
    return train_cfg


def resolve_resume_config_path(raw_path: Path) -> Path:
    path = raw_path.expanduser()
    if path.is_dir():
        candidates = [
            path / TRAIN_CONFIG_NAME,
            path / "pretrained_model" / TRAIN_CONFIG_NAME,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find {TRAIN_CONFIG_NAME} inside {path}. "
                "Provide the full path to the train config."
            )
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    return path.resolve()


def ensure_config_path_cli_arg(config_path: Path) -> None:
    cli_arg = f"--config_path={config_path}"
    if not any(arg.startswith("--config_path=") for arg in sys.argv[1:]):
        sys.argv.append(cli_arg)


def build_resume_config(config_path: Path) -> TrainWithCriticPipelineConfig:
    resolved_path = resolve_resume_config_path(config_path)
    ensure_config_path_cli_arg(resolved_path)
    cfg = TrainWithCriticPipelineConfig.from_pretrained(str(resolved_path))
    cfg.resume = True
    logging.info("Resuming training from %s", resolved_path)
    return cfg


def main() -> None:
    init_logging()
    args = parse_args()
    if args.resume:
        if args.config_path is None:
            raise ValueError("--config-path must be provided when using --resume.")
        train_cfg = build_resume_config(args.config_path)
    else:
        train_cfg = build_train_config(args)
    train_cfg.critic_only = args.critic_only
    train_cfg.log_policy_to_wandb = args.wandb_upload_policy
    train_cfg.log_code_to_wandb = args.wandb_upload_code

    train_cfg.validate()
    lerobot_train(train_cfg)

if __name__ == "__main__":
    main()
