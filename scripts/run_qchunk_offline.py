"""Train VGAS or VGAS+ with a shared critic-first pipeline."""

import argparse
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import draccus


DEFAULT_POLICY_PATH = Path(
    ""
)
DEFAULT_TEACHER_POLICY_PATH: Optional[Path] = None

DEFAULT_DATASET_ROOT = Path("")
DEFAULT_DATASET_REPO_ID = "libero_object"
DEFAULT_JOB_NAME: Optional[str] = None
DEFAULT_WANDB_PROJECT = "my_project"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    # Keep local lerobot as fallback path, do not shadow the active environment package.
    sys.path.append(str(LEROBOT_SRC))

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
        DistillConfig,
        TrainWithCriticPipelineConfig,
        train_from_config as lerobot_train,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import LeRobot dependencies. Ensure lerobot is installed or cloned alongside this repository. "
        f"Original error: {exc}"
    ) from exc


def _filter_smolvla_config_payload(payload: dict, *, source: str) -> dict:
    """Drop config keys unsupported by the active SmolVLAConfig version."""
    valid_keys = set(getattr(SmolVLAConfig, "__dataclass_fields__", {}).keys())
    if not valid_keys:
        valid_keys = set(inspect.signature(SmolVLAConfig).parameters.keys())
    dropped = sorted(key for key in payload if key not in valid_keys)
    if dropped:
        logging.warning(
            "Ignoring unsupported SmolVLA config keys from %s: %s",
            source,
            ", ".join(dropped),
        )
    return {key: value for key, value in payload.items() if key in valid_keys}


def parse_args() -> argparse.Namespace:
    def str2bool(value: str) -> bool:
        value = value.lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError("Expected a boolean value.")

    parser = argparse.ArgumentParser(
        description="Train VGAS (critic only) or VGAS+ (critic warmup followed by policy distillation)."
    )
    # Paths & dataset.
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH, help="Pretrained checkpoint to finetune.")
    parser.add_argument(
        "--teacher-policy-path",
        type=Path,
        default=DEFAULT_TEACHER_POLICY_PATH,
        help="Teacher pretrained checkpoint directory (defaults to --policy-path).",
    )
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional SmolVLA config.json to load (defaults to <policy-path>/config.json).")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Local dataset root if using local storage.")
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID, help="LeRobot dataset repo identifier.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Optional subset of episode indices.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"), help="Base directory for outputs.")
    parser.add_argument(
        "--output-dir-layout",
        type=str,
        default="legacy",
        choices=["legacy", "tag", "raw"],
        help="Output directory layout: legacy=outputs/train/<date>/<job>, tag=outputs/train/.../tag/<job>, raw=use --output-dir as-is.",
    )
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="Custom run name (defaults to q_agg_* auto naming when omitted).")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--config-path", type=Path, default=None, help=f"Path to an existing `{TRAIN_CONFIG_NAME}` when using `--resume`.")
    parser.add_argument(
        "--critic-init-checkpoint",
        type=Path,
        default=None,
        help="Optional phase-1 checkpoint dir / critic_pretrained_model dir / last.ckpt used to initialize the critic before stage-2.",
    )

    # Training schedule & runtime.
    parser.add_argument("--steps", type=int, default=12000, help="Number of optimisation steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for policy and data.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for training components.")
    parser.add_argument("--log-interval", type=int, default=200, help="Logging frequency in steps.")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, help="Checkpoint frequency in steps.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluation frequency during training (0 disables).")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming dataset loading mode.")
    parser.add_argument("--allow-missing-reward", type=str2bool, default=True, help="Allow datasets without reward/terminal by filling zeros.")
    parser.add_argument("--expert-width-multiplier", type=float, default=None, help="Width multiplier for the LM/expert layers (must match checkpoint).")
    parser.add_argument("--disable-save-checkpoint", action="store_true", help="Disable checkpoint saving even if LeRobot would normally save.")
    parser.add_argument("--policy-lr", type=float, default=None, help="Override SmolVLA policy optimizer peak LR.")
    parser.add_argument(
        "--policy-scheduler-warmup-steps",
        type=int,
        default=None,
        help="Override SmolVLA policy scheduler warmup steps.",
    )
    parser.add_argument(
        "--policy-scheduler-decay-steps",
        type=int,
        default=None,
        help="Override SmolVLA policy scheduler decay steps.",
    )
    parser.add_argument(
        "--policy-scheduler-decay-lr",
        type=float,
        default=None,
        help="Override SmolVLA policy scheduler final decay LR.",
    )

    # Policy chunking & rollout shapes.
    parser.add_argument("--chunk-size", type=int, default=32, help="Action chunk size for SmolVLA.")
    parser.add_argument("--n-action-steps", type=int, default=20, help="Number of supervised action steps.")
    parser.add_argument("--q-chunk-len", type=int, default=32, help="Critic/future-observation chunk length (defaults to --n-action-steps).")
    parser.add_argument("--discount", type=float, default=None, help="Discount factor (used for critic and mc returns).")
    parser.add_argument("--obs-steps", type=int, default=1, help="Number of observation steps provided to the model.")
    parser.add_argument(
        "--teacher-num-steps",
        type=int,
        default=None,
        help="Optional teacher flow-matching denoising steps override.",
    )
    parser.add_argument(
        "--student-num-steps",
        type=int,
        default=None,
        help="Optional student flow-matching denoising steps override.",
    )
    parser.add_argument(
        "--distill-phase1-steps",
        type=int,
        default=5000,
        help="Number of initial steps for phase-1 (teacher OOD + teacher-guided distillation).",
    )
    parser.add_argument(
        "--phase1-teacher-action-samples",
        type=int,
        default=1,
        help="Number of teacher action samples in phase-1 before critic best-of-N re-ranking.",
    )
    parser.add_argument(
        "--phase1-train-student",
        type=str2bool,
        default=False,
        help="Whether to update student in phase-1. Set false to make phase-1 critic-only warmup.",
    )
    parser.add_argument(
        "--phase1-use-student-as-source",
        type=str2bool,
        default=False,
        help="Use student policy (instead of teacher) as OOD/target source in phase-1.",
    )
    parser.add_argument(
        "--phase1-critic-updates-per-step",
        type=int,
        default=None,
        help="Critic updates per step in phase-1. Defaults to --critic-updates-per-step if not set.",
    )
    parser.add_argument(
        "--phase2-global-samples",
        type=int,
        default=8,
        help="Number of student candidate chunks sampled for phase-2 global optimization.",
    )
    parser.add_argument(
        "--phase2-local-samples",
        type=int,
        default=4,
        help="Number of global candidates kept for local optimization in phase-2.",
    )
    parser.add_argument(
        "--phase2-include-dataset-action-in-global-pool",
        action="store_true",
        help="Ablation: include dataset action chunk into phase-2 global candidate pool.",
    )
    parser.add_argument(
        "--phase2-local-opt-steps",
        type=int,
        default=5,
        help="Number of local action-gradient optimization steps in phase-2.",
    )
    parser.add_argument(
        "--phase2-local-opt-lr",
        type=float,
        default=3e-4,
        help="Step size used for local action optimization in phase-2.",
    )
    parser.add_argument(
        "--phase2-local-grad-normalize",
        type=str2bool,
        default=True,
        help="Normalize local Q-gradient per candidate chunk before updating actions.",
    )
    parser.add_argument(
        "--phase2-local-adv-weight",
        type=str2bool,
        default=False,
        help="Use tanh(z-score(Q)) as a per-candidate local update weight.",
    )
    parser.add_argument(
        "--phase2-local-adv-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for local gradient/advantage normalization.",
    )
    parser.add_argument(
        "--phase2-use-teacher-anchor",
        type=str2bool,
        default=True,
        help="Whether to include teacher-anchor distillation target in phase-2 actor updates.",
    )
    parser.add_argument(
        "--phase2-loss-rank-weight",
        type=float,
        default=None,
        help="Optional critic OOD rank loss weight for phase-2; defaults to --loss-rank-weight.",
    )
    parser.add_argument(
        "--distill-lambda-w2",
        type=float,
        default=0.2,
        help="Weight for teacher-anchor distillation loss in phase-2.",
    )
    parser.add_argument(
        "--distill-lambda-opt",
        type=float,
        default=1.0,
        help="Weight for optimized-action distillation loss in phase-2.",
    )
    parser.add_argument(
        "--mask-padded-action-loss",
        type=str2bool,
        default=True,
        help="Student-only: ignore padded action dims (> real action dim) when computing policy loss.",
    )
    parser.add_argument(
        "--phase2-disable-critic-best-of-n",
        dest="phase2_disable_critic_best_of_n",
        action="store_true",
        help="Disable best-of-n in critic target backup during phase-2 (qc-fql style).",
    )
    parser.add_argument(
        "--phase2-keep-critic-best-of-n",
        dest="phase2_disable_critic_best_of_n",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(phase2_disable_critic_best_of_n=False)
    parser.add_argument(
        "--distill",
        action="store_true",
        default=None,
        help="Train VGAS+: warm up the critic, then continue critic training while distilling the policy. Omit for VGAS.",
    )

    # Encoding/augmentation toggles.
    parser.add_argument("--use-data-augmentations", default=False, type=bool, help="Enable or disable visual data augmentations in encode_policy_observations_test.")
    parser.add_argument(
        "--use-vlm-backbone-encode",
        type=str2bool,
        nargs="?",
        const="true",
        default=True,
        help="Pass embeddings through VLM backbone encode in critic observation encoding.",
    )
    parser.add_argument(
        "--no-use-vlm-backbone-encode",
        dest="use_vlm_backbone_encode",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--load-vlm-weights", default=True, type=bool, help="Load the VLM backbone weights.")
    parser.add_argument("--unfreeze-vision-encoder", action="store_true", help="Finetune the vision encoder layers.")
    parser.add_argument("--train-full-model", action="store_true", help="Disable expert-only training.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push checkpoints to the Hugging Face Hub.")
    parser.add_argument("--policy-repo-id", type=str, default=None, help="Hub repo id when pushing to the hub.")

    # W&B.
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/organization.")
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="W&B logging mode override.")
    parser.add_argument("--wandb-notes", type=str, default=None, help="Optional notes attached to W&B run.")
    parser.add_argument("--wandb-upload-policy", dest="wandb_upload_policy", action="store_true", help="Upload policy checkpoints to W&B artifacts.")
    parser.add_argument("--no-wandb-upload-policy", dest="wandb_upload_policy", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--wandb-upload-code", dest="wandb_upload_code", action="store_true", help="Upload repository source code to W&B once at run start.")
    parser.add_argument("--no-wandb-upload-code", dest="wandb_upload_code", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(wandb_upload_policy=False, wandb_upload_code=False)

    # Environment helpers.
    parser.add_argument("--env-type", type=str, default="libero", help="Environment type to attach (use 'none' to disable, e.g. 'libero').")
    parser.add_argument("--env-task", type=str, default="libero_object", help="Libero task identifier when env-type=libero.")
    parser.add_argument("--env-obs-type", type=str, default="pixels_agent_pos", help="Observation type for Libero env (pixels or pixels_agent_pos).")
    parser.add_argument("--env-camera-name", type=str, default="agentview_image,robot0_eye_in_hand_image", help="Comma-separated camera names for Libero env.")
    parser.add_argument("--env-fps", type=int, default=30, help="Frame rate for Libero env.")
    parser.add_argument("--env-episode-length", type=int, default=520, help="Episode length for Libero env.")
    parser.add_argument("--env-disable-init-states", action="store_true", help="Disable loading stored initial states for the Libero environment.")

    # Critic training knobs.
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512], help="Hidden dimensions for the critic backbone MLP.")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Critic optimizer learning rate.")
    parser.add_argument("--critic-betas", type=float, nargs=2, default=[0.9, 0.95], metavar=("BETA1", "BETA2"), help="Critic Adam betas.")
    parser.add_argument("--critic-weight-decay", type=float, default=1e-10, help="Weight decay applied to critic parameters.")
    parser.add_argument("--critic-tau", type=float, default=0.005, help="Soft target update coefficient.")
    parser.add_argument("--critic-tau-warmup", type=float, default=None, help="Optional tau used during warmup steps before switching to --critic-tau.")
    parser.add_argument("--critic-tau-warmup-steps", type=int, default=0, help="Number of initial steps to use warmup tau (if provided).")
    parser.add_argument("--critic-grad-clip", type=float, default=10.0, help="Gradient clipping norm for critic updates.")
    parser.add_argument("--critic-grad-clip-warmup", type=float, default=None, help="Optional gradient clip to use during warmup steps (uses --ood-warmup-steps as boundary).")
    parser.add_argument(
        "--critic-updates-per-step",
        type=int,
        default=1,
        help="Number of critic updates per training step (actor update remains once per step).",
    )
    parser.add_argument(
        "--critic-backup-action-samples",
        "--critic-action-samples",
        dest="critic_backup_action_samples",
        type=int,
        default=8,
        help="Number of candidate chunks used in critic Expected-Max backup (TD target).",
    )
    parser.add_argument("--critic-q-agg", type=str, default="min", choices=["mean", "min", "max"], help="Aggregation used when combining twin Q estimates.")
    parser.add_argument("--critic-att-mode", type=str, default="bi-level", choices=["causal", "bi-level"], help="Attention pattern for transformer-based critic heads.")
    parser.add_argument("--critic-temperature", type=float, default=2.0, help="Noise scale for candidate actions.")
    parser.add_argument("--critic-mask-dropout-prob", type=float, default=0.5, help="Dropout prob for action padding mask during critic training (0 disables).")
    parser.add_argument("--critic-value-head-bias-enable", action="store_true", help="Initialize value head final-layer bias to a constant.")
    parser.add_argument("--critic-value-head-bias-value", type=float, default=0.0, help="Constant bias value when --critic-value-head-bias-enable is set.")
    parser.add_argument("--critic-use-dual-noise-ood", action="store_true", help="Use dual-noise GT negatives (tiny + small noise) instead of truncation OOD.")
    parser.add_argument("--critic-action-weights", type=float, nargs="+", default=None, help="Per-dimension weights for action distance in OOD penalty (length must match action dim).")
    parser.add_argument("--critic-warmup-steps", type=int, default=1000, help="Warmup steps for the critic learning rate.")
    parser.add_argument(
        "--critic-total-steps",
        type=int,
        default=None,
        help="Optional total steps for critic LR scheduling (defaults to --steps).",
    )
    parser.add_argument("--critic-lr-final", type=float, default=2.5e-06, help="Final critic learning rate after decay (default 0, i.e. decay to zero).")
    parser.add_argument(
        "--critic-scheduler-step-mode",
        type=str,
        default="critic_update",
        choices=["critic_update", "outer_step"],
        help=(
            "How to advance the critic LR scheduler: every critic optimizer update "
            "or once per outer training step."
        ),
    )

    # Critic architecture.
    parser.add_argument("--critic-type", type=str, default="q_chunk_former", choices=["mlp", "q_chunk_former"], help="Critic architecture to use for QC training (legacy names map to q_chunk_former).")
    parser.add_argument("--critic-head-type", type=str, default="transformer", choices=["mlp", "transformer"], help="Head to pair with value_query_head critics.")
    parser.add_argument("--critic-head-num-layers", type=int, default=2, help="Number of transformer layers for the critic head when applicable.")
    parser.add_argument("--critic-head-mlp-dims", type=int, nargs="+", default=None, help="MLP dimensions used inside the critic head (defaults to critic hidden dims).")
    parser.add_argument("--critic-vqh-hidden-dims", type=int, nargs="+", default=None, help="Hidden dims for ValueQueryHead backbones (defaults to critic hidden dims).")
    parser.add_argument(
        "--critic-qformer-num-backbone-layers",
        type=int,
        default=2,
        help="Number of decoder layers for QFormer backbones.",
    )
    parser.add_argument("--critic-vqh-vlm-model-name", type=str, default=None, help="Override the VLM model name used when instantiating ValueQueryHead backbones.")
    parser.add_argument("--critic-num-q-heads", type=int, default=2, help="Number of independent Q heads when using ValueQueryHead critics.")
    parser.add_argument("--critic-loss-mode", type=str, default="per_head_mean", choices=["mse", "per_head_mean"], help="Reduction applied to critic TD errors.")
    # Raw state fusion.
    parser.add_argument("--use-raw-state-fusion", type=str2bool, default=True, help="Enable raw state fusion into critic action embeddings (requires observation.state).")
    parser.add_argument(
        "--critic-use-state-encoder",
        type=str2bool,
        default=False,
        help="Use a critic-side independent state encoder initialized from student state_proj.",
    )
    parser.add_argument(
        "--critic-use-independent-encoder",
        type=str2bool,
        default=False,
        help="Use an independent critic encoder stack (vision+language+state) without actor VLM forward.",
    )
    parser.add_argument(
        "--critic-vision-freeze-layers",
        type=int,
        default=8,
        help="Number of bottom vision encoder layers to freeze when critic independent encoder is enabled.",
    )
    parser.add_argument("--raw-state-dim", type=int, default=8, help="Dimension of observation.state when raw state fusion is enabled.")
    # parser.add_argument("--num-query-token", type=int, default=16, help="Number of query tokens for transformer critic heads.")
    # parser.add_argument("--use-query-head", type=str2bool, dest="critic_query_head", default=False, help="Whether to use the query-based critic head (set False to use the no-query variant).")
    parser.add_argument("--critic-value-head-num-layers", type=int, default=2, help="Number of transformer layers for value_head critics.")
    parser.add_argument("--critic-value-head-mlp-dims", type=int, nargs="+", default=None, help="MLP hidden dims for value_head critics (defaults to critic hidden dims).")
    parser.add_argument("--critic-value-head-vlm-model-name", type=str, default=None, help="Override VLM model when constructing value_head critics.")

    # CalQL & OOD regularization.
    parser.add_argument("--use-calql", type=str2bool, default=False, help="Enable CalQL regularization in critic (True/False).")
    parser.add_argument("--use-ood-reg", type=str2bool, default=True, help="Enable explicit OOD penalty regularization.")
    parser.add_argument("--ood-alpha", type=float, default=2.0, help="Weight for OOD regularization term.")
    parser.add_argument("--ood-action-source", type=str, default="erg", choices=["erg", "cql"], help="OOD action source ('erg' or 'cql').")
    parser.add_argument("--dist-penalty-beta", type=float, default=5, help="Slope for distance-based OOD target.")
    parser.add_argument("--dist-clamp-max", type=float, default=None, help="Optional clamp on OOD distance target.")
    parser.add_argument("--ood-warmup-steps", type=int, default=0, help="Warmup steps before enabling OOD regularization.")
    parser.add_argument("--ood-include-current-actions", type=str2bool, default=True, help="Include policy current actions when building OOD samples.")
    parser.add_argument("--ood-include-random-actions", type=str2bool, default=False, help="Include random/noise actions when building OOD samples.")
    parser.add_argument("--ood-include-next-actions", type=str2bool, default=False, help="Include next-state actions when building OOD samples (CalQL forces True).")
    parser.add_argument("--use-ood-noise", type=str2bool, default=True, help="Add Gaussian noise to OOD actions.")
    parser.add_argument("--use-ood-trunc", type=str2bool, default=True, help="Include truncated actions in OOD pool.")
    parser.add_argument("--use-ood-mix", type=str2bool, default=False, help="Include mixed actions in OOD pool.")
    parser.add_argument("--ood-noise-stds", type=float, nargs="+", default=[0.02], help="Noise stds for OOD action perturbations.")
    parser.add_argument("--ood-mix-ratio", type=float, default=0.5, help="Fraction of mixed actions to include in OOD pool.")
    parser.add_argument("--ood-mix-alpha-low", type=float, default=0.3, help="Low alpha for action mixing.")
    parser.add_argument("--ood-mix-alpha-high", type=float, default=0.7, help="High alpha for action mixing.")
    parser.add_argument("--debug-mix-dist", type=str2bool, default=False, help="Log debug stats for OOD mixing distances.")
    parser.add_argument("--loss-anchor-weight", type=float, default=1.0, help="Weight for anchor OOD loss term.")
    parser.add_argument("--loss-rank-weight", type=float, default=1.0, help="Weight for pairwise OOD loss term.")
    parser.add_argument("--ood-m-actions",dest="ood_m_actions",type=int,default=2,help="Number of OOD action samples (defaults to all best-of-n candidates).")
    parser.add_argument("--cql-alpha", type=float, default=1.0, help="Weight for CalQL/CQL regularization term.")
    parser.add_argument("--cql-next-noise-std", type=float, default=0.05, help="Std for CalQL next-action noise (set 0 to disable).")
    parser.add_argument("--cql-cur-noise-std", type=float, default=None, help="Std for CalQL current-action noise (defaults to next noise std; set 0 to disable).")

    return parser.parse_args()


def build_job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    if not args.distill:
        return f"vgas_{args.dataset_repo_id}"
    teacher_steps = args.teacher_num_steps if args.teacher_num_steps is not None else "cfg"
    student_steps = args.student_num_steps if args.student_num_steps is not None else "cfg"
    return f"vgas_plus_t{teacher_steps}_s{student_steps}"


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
    if args.critic_type in {"my_value_query_head", "my_value_head", "value_head"}:
        args.critic_type = "q_chunk_former"
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
        enable=True,
        hidden_dims=tuple(args.critic_hidden_dims),
        lr=args.critic_lr,
        betas=(args.critic_betas[0], args.critic_betas[1]),
        weight_decay=args.critic_weight_decay,
        discount=(args.discount if args.discount is not None else 0.98),
        tau=args.critic_tau,
        tau_warmup=args.critic_tau_warmup,
        tau_warmup_steps=args.critic_tau_warmup_steps,
        grad_clip_norm=args.critic_grad_clip,
        grad_clip_warmup=args.critic_grad_clip_warmup,
        critic_updates_per_step=max(1, int(args.critic_updates_per_step)),
        action_samples=args.critic_backup_action_samples,
        backup_action_samples=args.critic_backup_action_samples,
        q_aggregation=args.critic_q_agg,
        temperature=args.critic_temperature,
        critic_type=args.critic_type,
        qformer_num_backbone_layers=args.critic_qformer_num_backbone_layers,
        vqh_hidden_dims=vqh_hidden_dims,
        vqh_vlm_model_name=args.critic_vqh_vlm_model_name,
        head_type=args.critic_head_type,
        head_num_layers=args.critic_head_num_layers,
        head_mlp_dims=head_mlp_dims,
        num_q_heads=args.critic_num_q_heads,
        critic_loss_mode=args.critic_loss_mode,
        att_mode=args.critic_att_mode,
        # num_query_token=args.num_query_token,
        value_head_num_layers=args.critic_value_head_num_layers,
        value_head_mlp_dims=value_head_mlp_dims,
        value_head_vlm_model_name=args.critic_value_head_vlm_model_name,
        lr_warmup_steps=args.critic_warmup_steps,
        lr_total_steps=args.critic_total_steps or args.steps,
        lr_final=args.critic_lr_final,
        scheduler_step_mode=args.critic_scheduler_step_mode,
        use_calql=args.use_calql,
        ood_m_actions=args.ood_m_actions,
        cql_alpha=args.cql_alpha,
        cql_next_noise_std=args.cql_next_noise_std,
        cql_cur_noise_std=(
            args.cql_cur_noise_std if args.cql_cur_noise_std is not None else args.cql_next_noise_std
        ),
        # use_no_query_head=not args.critic_query_head,
        use_ood_reg=args.use_ood_reg,
        ood_alpha=args.ood_alpha,
        ood_action_source=args.ood_action_source,
        dist_penalty_beta=args.dist_penalty_beta,
        dist_clamp_max=args.dist_clamp_max,
        ood_warmup_steps=args.ood_warmup_steps,
        ood_include_current_actions=args.ood_include_current_actions,
        ood_include_random_actions=args.ood_include_random_actions,
        ood_include_next_actions=args.ood_include_next_actions,
        ood_noise_stds=tuple(args.ood_noise_stds),
        use_ood_noise=args.use_ood_noise,
        use_ood_trunc=args.use_ood_trunc,
        use_ood_mix=args.use_ood_mix,
        ood_mix_ratio=args.ood_mix_ratio,
        ood_mix_alpha_low=args.ood_mix_alpha_low,
        ood_mix_alpha_high=args.ood_mix_alpha_high,
        debug_mix_dist=args.debug_mix_dist,
        loss_anchor_weight=args.loss_anchor_weight,
        loss_rank_weight=args.loss_rank_weight,
        use_raw_state_fusion=args.use_raw_state_fusion,
        use_vlm_backbone_encode=args.use_vlm_backbone_encode,
        use_critic_state_encoder=args.critic_use_state_encoder,
        use_independent_critic_encoder=args.critic_use_independent_encoder,
        critic_vision_freeze_layers=max(0, int(args.critic_vision_freeze_layers)),
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
    output_dir_layout = getattr(args, "output_dir_layout", "legacy")
    if output_dir_layout == "legacy":
        job_name += "test"
        date_prefix = datetime.now().strftime("%m.%d")
        output_dir = args.output_dir / date_prefix / job_name
    elif output_dir_layout == "tag":
        output_dir = args.output_dir / job_name
    elif output_dir_layout == "raw":
        output_dir = args.output_dir
    else:
        raise ValueError(f"Unknown output_dir_layout '{output_dir_layout}'.")

    n_action_steps = args.n_action_steps or args.chunk_size
    if n_action_steps > args.chunk_size:
        raise ValueError(f"`n_action_steps` ({n_action_steps}) cannot exceed `chunk_size` ({args.chunk_size}).")
    q_chunk_len = args.q_chunk_len or n_action_steps
    if q_chunk_len > args.chunk_size:
        raise ValueError(f"`q_chunk_len` ({q_chunk_len}) cannot exceed `chunk_size` ({args.chunk_size}).")

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
            "load_vlm_weights": args.load_vlm_weights,
            "freeze_vision_encoder": not args.unfreeze_vision_encoder,
            "train_expert_only": not args.train_full_model,
        }
    )
    if args.student_num_steps is not None:
        policy_cfg_payload["num_steps"] = int(args.student_num_steps)
    if args.expert_width_multiplier is not None:
        policy_cfg_payload["expert_width_multiplier"] = args.expert_width_multiplier
    if args.policy_lr is not None:
        policy_cfg_payload["optimizer_lr"] = float(args.policy_lr)
    if args.policy_scheduler_warmup_steps is not None:
        policy_cfg_payload["scheduler_warmup_steps"] = int(args.policy_scheduler_warmup_steps)
    if args.policy_scheduler_decay_steps is not None:
        policy_cfg_payload["scheduler_decay_steps"] = int(args.policy_scheduler_decay_steps)
    if args.policy_scheduler_decay_lr is not None:
        policy_cfg_payload["scheduler_decay_lr"] = float(args.policy_scheduler_decay_lr)
    policy_cfg_payload = _filter_smolvla_config_payload(
        policy_cfg_payload,
        source=str(policy_config_path) if policy_config_path is not None else "<runtime overrides>",
    )
    policy_cfg = SmolVLAConfig(**policy_cfg_payload)

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root) if args.dataset_root else None,
        episodes=args.episodes,
        streaming=args.streaming,
    )
    dataset_cfg.allow_missing_reward = args.allow_missing_reward

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
    # Keep the critic and the dataset on the same discount factor.
    if args.discount is not None:
        critic_cfg.discount = args.discount
    if args.use_calql:
        critic_cfg.use_calql = True
    if args.use_ood_reg:
        critic_cfg.use_ood_reg = True

    teacher_path = args.teacher_policy_path or args.policy_path
    distill_cfg = DistillConfig(
        enable=bool(args.distill),
        teacher_pretrained_path=str(teacher_path) if teacher_path else None,
        teacher_num_steps=args.teacher_num_steps,
        student_num_steps=args.student_num_steps,
        mask_padded_action_loss=args.mask_padded_action_loss,
        phase1_teacher_action_samples=int(args.phase1_teacher_action_samples),
        action_samples=None,
        freeze_teacher=True,
        phase1_steps=args.distill_phase1_steps,
        phase1_train_student=args.phase1_train_student,
        phase1_use_student_as_source=args.phase1_use_student_as_source,
        phase1_critic_updates_per_step=args.phase1_critic_updates_per_step,
        phase2_global_samples=args.phase2_global_samples,
        local_samples=args.phase2_local_samples,
        phase2_include_dataset_action_in_global_pool=args.phase2_include_dataset_action_in_global_pool,
        phase2_local_opt_steps=args.phase2_local_opt_steps,
        phase2_local_opt_lr=args.phase2_local_opt_lr,
        phase2_local_grad_normalize=args.phase2_local_grad_normalize,
        phase2_local_adv_weight=args.phase2_local_adv_weight,
        phase2_local_adv_eps=args.phase2_local_adv_eps,
        phase2_disable_critic_best_of_n=args.phase2_disable_critic_best_of_n,
        phase2_use_teacher_anchor=args.phase2_use_teacher_anchor,
        phase2_loss_rank_weight=args.phase2_loss_rank_weight,
        lambda_w2=args.distill_lambda_w2,
        lambda_opt=args.distill_lambda_opt,
    )

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
        distill=distill_cfg,
        log_policy_to_wandb=args.wandb_upload_policy,
        log_code_to_wandb=args.wandb_upload_code,
        q_chunk_len=q_chunk_len,
        critic_init_checkpoint=args.critic_init_checkpoint,
    )
    # Propagate the discount to the custom dataset so both stay consistent.
    if args.discount is not None:
        train_cfg.dataset.discount = args.discount
    if args.use_calql:
        train_cfg.critic.use_calql = True
    # Toggles forwarded to encode_policy_observations_test
    train_cfg.use_vlm_backbone_encode = args.use_vlm_backbone_encode
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
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    critic_payload = payload.get("critic")
    distill_payload = payload.get("distill")
    if not isinstance(distill_payload, dict):
        distill_payload = {}
        payload["distill"] = distill_payload
    removed_fields = []
    if "critic_only" in payload:
        legacy_critic_only = bool(payload.pop("critic_only"))
        distill_payload["enable"] = not legacy_critic_only
        removed_fields.append("critic_only")
    if isinstance(critic_payload, dict):
        for key in (
            "eval_ranking_freq",
            "eval_ranking_batches",
            "eval_ranking_action_samples",
            "eval_ranking_batch_size",
            "eval_ranking_start_step",
            "eval_ranking_train",
            "eval_ranking_full",
            "eval_ranking_full_dataset_root",
        ):
            if key in critic_payload:
                critic_payload.pop(key)
                removed_fields.append(key)
    if removed_fields:
        logging.info("Migrating removed fields in resume config: %s", ", ".join(removed_fields))
        cfg = draccus.decode(TrainWithCriticPipelineConfig, payload)
    else:
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
    if args.resume and args.distill is not None and bool(train_cfg.distill.enable) != bool(args.distill):
        raise ValueError("Cannot change VGAS/VGAS+ mode while resuming; use the mode stored in train_config.json.")
    train_cfg.critic.enable = True
    train_cfg.log_policy_to_wandb = args.wandb_upload_policy
    train_cfg.log_code_to_wandb = args.wandb_upload_code

    train_cfg.validate()
    lerobot_train(train_cfg)

if __name__ == "__main__":
    main()
