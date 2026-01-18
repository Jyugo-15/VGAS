"""Offline similarity eval: GT actions vs policy best-of-n samples (optional critic-guided).

This script loads a pretrained SmolVLA (and optional best-of-n critic), iterates
over a LeRobot dataset, samples N candidate action chunks, and reports cosine
similarity between each candidate and the ground-truth action chunk.

It is standalone and does not modify existing training/eval code.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from data.lerobot_reward_dataset import RewardAugmentedLeRobotDataset  # noqa: E402
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from qchunk.best_of_n_critic import BestOfNCriticTrainer  # noqa: E402
from smolvla_qchunk.eval.bestofn_eval import (  # noqa: E402
    _load_policy_config,
    _load_critic_checkpoint,
    _resolve_critic_state_path,
)
from scripts.train_qchunk_offline import CriticConfig, encode_policy_observations  # noqa: E402

DEFAULT_ACTION_WEIGHTS: Sequence[float] = (5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0)


def _str2bool(v: Any) -> bool:
    """Parse common string/boolean inputs into a real bool for argparse."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lv = v.lower()
        if lv in ("yes", "true", "t", "1", "y", "on"):
            return True
        if lv in ("no", "false", "f", "0", "n", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity between GT actions and best-of-n SmolVLA samples."
    )
    #official path                    /home/chanxu/Data/workplace/vla_exp/lerobot/models/smolvla_libero 
    #mix_task_suit_fewshot            /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/offline_qc_target_discount_few_shot_full_data_20251117_233140/checkpoints/010000/pretrained_model 
    #object_few_shot                  /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/11.23/q_agg_min_layer_2_crit_lm_per_head_mean/checkpoints/010000/pretrained_model       
    parser.add_argument("--policy-path", type=Path, default="/home/chanxu/Data/workplace/vla_exp/lerobot/outputs/train/my_smolvla_new_5_shot/checkpoints/005000/pretrained_model", help="SmolVLA pretrained checkpoint directory.")
    #
    #official path critic             /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/offline_qc_target_discount_few_shot_full_data_20251117_233140/checkpoints/010000/critic_pretrained_model/last.ckpt
    #mix_task_suit_fewshot            /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/q_only_single_suit_pretrained_policy_10_20251120_002752/checkpoints/010000/critic_pretrained_model/last.ckpt
    #object_few_shot                  /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/11.23/q_agg_mean_layer_2_crit_lm_mse/checkpoints/010000/critic_pretrained_model/last.ckpt
    # #                               /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/11.23/q_agg_mean_layer_2_crit_lm_per_head_mean/checkpoints/010000/critic_pretrained_model/last.ckpt       
    # #                               /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/11.23/q_agg_min_layer_2_crit_lm_per_head_mean/checkpoints/010000/critic_pretrained_model/last.ckpt
    # #                               /home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/11.23/q_agg_min_layer_4_crit_lm_per_head_mean/checkpoints/010000/critic_pretrained_model/last.ckpt
    parser.add_argument(
        "--critic-state", type=Path, default="/home/chanxu/Data/workplace/vla_exp/smolvla_qchunk/outputs/train/12.06/calql_fusion_no_query_5000_obj_chage_dis_aggtest/checkpoints/005000/critic_pretrained_model/last.ckpt", help="Critic checkpoint dir or file (optional)."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/simanalyse/12-06/calql_no_query_5000_chage_dis_agg_fowshot_5000_max.json"),
        help="Where to save aggregate metrics.",
    )

    #全数据                           /home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero
    #object-fewshot                  /home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero_with_rewards_5shot/libero_object
    #object-full                     /home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero_split/libero_object
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="/home/chanxu/Data/workplace/vla_exp/lerobot/dataset_news/HF_LIBERO_5SHORT/libero_object",
        # /home/chanxu/Data/workplace/vla_exp/lerobot/dataset/HFlibero_split/libero_object
        # /home/chanxu/Data/workplace/vla_exp/lerobot/dataset_news/HF_LIBERO_5SHORT/libero_object
        help="Local LeRobot dataset root (same format used for training).",
    )
    parser.add_argument("--dataset-repo-id", type=str, default="libero_object", help="Dataset repo id/name.")
    parser.add_argument("--episodes", type=int, nargs="+", default=[1,2,3], help="Optional subset of episode indices.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=16, help="Dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--action-samples", type=int, default=8, help="Number of candidate samples (N).")
    parser.add_argument(
        "--critic-q-agg",
        type=str,
        default="max",
        choices=["mean", "min", "max"],
        help="Override critic q_aggregation during similarity eval (defaults to checkpoint value).",
    )
    parser.add_argument(
        "--use-current-critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use current critic (not target) for scoring candidates.",
    )

    parser.add_argument(
        "--use-data-augmentations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mark whether data augmentations are used (for naming only).",
    )
    parser.add_argument(
        "--use-calql",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mark whether CalQL is used (for naming only).",
    )
    parser.add_argument(
        "--use-postprocess-sim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply postprocessor before computing similarity (default: True). Disable to compare raw model outputs.",
    )
    parser.add_argument(
        "--action-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTION_WEIGHTS),
        help="Per-action-dimension weights for weighted similarity metrics (length must match action_dim).",
    )
    parser.add_argument(
        "--use-critic-weights",
        type=_str2bool,
        default=True,
        metavar="{True,False}",
        help="If a critic checkpoint has action_distance_weights, use them for weighted similarity.",
    )
    parser.add_argument(
        "--sim-reduction",
        type=str,
        choices=["flatten", "per_dim_mean", "weighted", "both"],
        default="weighted",
        help="Similarity reduction: cosine flatten, per-action-dim cosine, weighted L2 (negative distance), or compute both cosine variants.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=32,
        help="Override policy n_action_steps/chunk length (defaults to policy config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling/shuffling.",
    )
    parser.add_argument(
        "--limit-steps",
        type=int,
        default=64,
        help="Max number of dataloader steps to evaluate (default 2 for quick random sampling).",
    )

    parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming mode if supported.")
    return parser.parse_args()


def build_policy_bundle(
    policy_cfg: Any,
    policy_path: Path,
    device: str,
    ds_meta: Any,
) -> Tuple[Any, Any, Any]:
    """Instantiate policy + processors using dataset metadata for feature shapes/stats."""
    policy_cfg = policy_cfg
    policy_cfg.device = device
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, env_cfg=None)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
        dataset_stats=getattr(ds_meta, "stats", None),
    )
    return policy, preprocessor, postprocessor


def build_critic(
    critic_state_path: Optional[Path],
    policy: Any,
    batch: Dict[str, torch.Tensor],
    device: str,
    action_samples: int,
    critic_q_agg_override: Optional[str] = None,
) -> Optional[BestOfNCriticTrainer]:
    if critic_state_path is None:
        return None
    resolved = _resolve_critic_state_path(critic_state_path)
    if resolved is None:
        logging.warning("Could not resolve critic checkpoint at %s", critic_state_path)
        return None
    ckpt = _load_critic_checkpoint(resolved)
    cfg = ckpt.config if isinstance(ckpt.config, CriticConfig) else CriticConfig()

    cfg.action_samples = max(action_samples, 1)
    if critic_q_agg_override:
        cfg.q_aggregation = critic_q_agg_override
    print("*****************************")
    print(cfg.q_aggregation)
    trainer = BestOfNCriticTrainer.build(
        policy=policy,
        batch=batch,
        cfg=cfg,
        device=torch.device(device),
        encoder_fn=encode_policy_observations,
    )
    trainer.load_state_dict(ckpt.state_dict)
    trainer.critic.eval()
    trainer.target_critic.eval()
    return trainer


def _repeat_batch(batch: Dict[str, Any], repeats: int) -> Dict[str, Any]:
    if repeats <= 1:
        return batch
    first_tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
    if first_tensor is None:
        return batch
    bs = first_tensor.shape[0]
    expanded: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == bs:
            expanded[k] = torch.repeat_interleave(v, repeats, dim=0)
        else:
            expanded[k] = v
    return expanded


def _get_pad_mask(batch: Dict[str, Any], chunk_size: int) -> Optional[torch.Tensor]:
    """Extract per-timestep pad mask if present; returns shape (bs, chunk)."""
    mask = batch.get("actions_is_pad")
    if mask is None and "masks" in batch:
        # masks is 1 for valid steps; invert to get pad
        m = batch["masks"]
        mask = ~m.to(torch.bool)
    if mask is None:
        return None
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return mask[:, :chunk_size]


def sample_candidates(
    policy: Any,
    batch: Dict[str, torch.Tensor],
    critic_trainer: Optional[BestOfNCriticTrainer],
    action_samples: int,
    device: str,
    chunk_size: int,
    use_current_critic: bool = False,
    forced_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return candidates (bs, n, chunk, act_dim), optional q (bs, n, 1), optional best_q."""
    action_samples = max(action_samples, 1)
    encoding = encode_policy_observations(policy, batch).to(device)
    first_tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
    if first_tensor is None:
        raise ValueError("Batch does not contain tensor observations.")
    batch_size = first_tensor.shape[0]
    if action_samples == 1:
        actions = policy.predict_action_chunk(batch)
        actions = actions[:, :chunk_size]
        return actions.unsqueeze(1), None, None

    expanded_batch = _repeat_batch(batch, action_samples)
    with torch.no_grad():
        expanded_actions = policy.predict_action_chunk(expanded_batch)
    if expanded_actions.shape[1] < chunk_size:
        raise ValueError(f"Policy chunk {expanded_actions.shape[1]} shorter than requested {chunk_size}.")
    expanded_actions = expanded_actions[:, :chunk_size]
    act_dim = expanded_actions.shape[-1]
    candidates = expanded_actions.view(batch_size, action_samples, chunk_size, act_dim)

    if critic_trainer is None:
        return candidates, None, None

    critic_model = critic_trainer.critic if use_current_critic else critic_trainer.target_critic
    flat_actions = candidates.view(-1, chunk_size, act_dim)
    repeated_encoding = encoding.repeat(action_samples)
    if forced_mask is not None:
        mask = forced_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask[:, :chunk_size].to(device=device, dtype=torch.bool)
        mask_to_use = mask.unsqueeze(1).repeat(1, action_samples, 1).view(-1, chunk_size)
    else:
        mask_to_use = torch.zeros(flat_actions.shape[:2], dtype=torch.bool, device=device)
    q_values = critic_model(repeated_encoding, flat_actions, action_mask=mask_to_use)
    q = critic_trainer._aggregate_q(*q_values)  # shape (bs*n, 1)
    q = q.view(batch_size, action_samples, -1)
    best_indices = torch.argmax(q, dim=1).squeeze(-1)
    batch_indices = torch.arange(batch_size, device=device)
    best_q = q[batch_indices, best_indices]
    return candidates, q, best_q


def cosine_sim(
    gt: torch.Tensor,
    candidates: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    reduction: str = "flatten",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    gt: (bs, chunk, dim); candidates: (bs, n, chunk, dim)
    pad_mask: (bs, chunk) True where padded.
    reduction:
        - "flatten": flatten (chunk, dim) then cosine.
        - "per_dim_mean": per action-dim cosine over time, then mean over dim.
    Returns (sim_mat (bs, n), valid_rows (bs,)).
    """
    bs, n, chunk, dim = candidates.shape
    if pad_mask is None:
        pad_mask = torch.zeros((bs, chunk), device=gt.device, dtype=torch.bool)
    pad_mask = pad_mask[:, :chunk]
    valid_mask = (~pad_mask).to(gt.dtype).unsqueeze(-1)  # (bs, chunk, 1)

    if reduction == "per_dim_mean":
        valid_rows = valid_mask.squeeze(-1).sum(dim=1) > 0
        masked_gt = gt[:, :chunk] * valid_mask
        masked_candidates = candidates * valid_mask.unsqueeze(1)
        dot = (masked_candidates * masked_gt.unsqueeze(1)).sum(dim=2)  # (bs, n, dim)
        gt_norm = masked_gt.pow(2).sum(dim=1).clamp_min(1e-8).sqrt()  # (bs, dim)
        cand_norm = masked_candidates.pow(2).sum(dim=2).clamp_min(1e-8).sqrt()  # (bs, n, dim)
        sim_per_dim = dot / (cand_norm * gt_norm.unsqueeze(1) + 1e-8)  # (bs, n, dim)
        sim = sim_per_dim.mean(dim=-1)
        return sim, valid_rows

    masked_gt = gt[:, :chunk] * valid_mask
    masked_candidates = candidates * valid_mask.unsqueeze(1)
    dot = (masked_candidates * masked_gt.unsqueeze(1)).sum(dim=(2, 3))  # bs * n_sample
    gt_norm = masked_gt.pow(2).sum(dim=(1, 2)).clamp_min(1e-8).sqrt()
    cand_norm = masked_candidates.pow(2).sum(dim=(2, 3)).clamp_min(1e-8).sqrt()
    sim = dot / (cand_norm * gt_norm.unsqueeze(1) + 1e-8)

    valid_rows = valid_mask.squeeze(-1).sum(dim=1) > 0
    return sim, valid_rows


def weighted_l2_sim(
    gt: torch.Tensor,
    candidates: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    weights_list: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Similarity based on negative weighted L2 distance (aligns with critic distance)."""
    bs, n, chunk, dim = candidates.shape
    if pad_mask is None:
        pad_mask = torch.zeros((bs, chunk), device=candidates.device, dtype=torch.bool)
    pad_mask = pad_mask[:, :chunk]
    valid_mask = (~pad_mask).to(candidates.dtype).unsqueeze(1).unsqueeze(-1)  # (bs,1,chunk,1)

    if weights_list is None:
        weights_list = DEFAULT_ACTION_WEIGHTS
    if len(weights_list) != dim:
        raise ValueError(f"weights_list length ({len(weights_list)}) must match action dim ({dim}).")
    weights = torch.tensor(weights_list, device=candidates.device, dtype=candidates.dtype).view(1, 1, 1, -1)

    diff = (candidates - gt.unsqueeze(1)) ** 2
    weighted = diff * weights
    dist_per_step = weighted.sum(dim=-1)  # (bs, n, chunk)
    valid_counts = valid_mask.squeeze(-1).sum(dim=-1).clamp_min(1.0)  # (bs, n)
    dist = (dist_per_step * valid_mask.squeeze(-1)).sum(dim=-1) / valid_counts  # (bs, n)
    sim = -dist  # higher is better (closer)
    valid_rows = valid_counts.sum(dim=1) > 0
    return sim, valid_rows


def pairwise_candidate_extremes(
    candidates: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    reduction: str = "flatten",
    weights_list: Optional[Sequence[float]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Return max/min pairwise cosine between candidate actions (per sample)."""
    bs, n, chunk, dim = candidates.shape
    if n < 2:
        return None, None, torch.zeros(bs, device=candidates.device, dtype=torch.bool)

    if pad_mask is None:
        pad_mask = torch.zeros((bs, chunk), device=candidates.device, dtype=torch.bool)
    pad_mask = pad_mask[:, :chunk]
    valid_mask = (~pad_mask).to(candidates.dtype).unsqueeze(-1)  # (bs, chunk, 1)

    if reduction == "per_dim_mean":
        masked = candidates * valid_mask.unsqueeze(1)
        # shape: bs, dim, n, chunk
        x = masked.permute(0, 3, 1, 2)
        dot = torch.einsum("bdic,bdjc->bdij", x, x)  # (bs, dim, n, n)
        norm = x.pow(2).sum(dim=3).clamp_min(1e-8).sqrt()  # (bs, dim, n)
        denom = norm.unsqueeze(2) * norm.unsqueeze(3)
        pairwise = dot / (denom + 1e-8)
        pairwise = pairwise.mean(dim=1)  # (bs, n, n)
    elif reduction == "weighted":
        # negative weighted distance as similarity
        if weights_list is None:
            weights_list = DEFAULT_ACTION_WEIGHTS
        if len(weights_list) != dim:
            raise ValueError(f"weights_list length ({len(weights_list)}) must match action dim ({dim}).")
        weights = torch.tensor(weights_list, device=candidates.device, dtype=candidates.dtype).view(1, 1, 1, 1, -1)
        mask = valid_mask.unsqueeze(1).unsqueeze(2)  # (bs,1,1,chunk,1)
        diff = candidates.unsqueeze(2) - candidates.unsqueeze(1)  # (bs, n, n, chunk, dim)
        dist_per_step = (diff ** 2 * weights).sum(dim=-1)  # (bs, n, n, chunk)
        valid_counts = mask.squeeze(-1).sum(dim=-1).clamp_min(1.0)  # (bs,1,1,chunk)->(bs,1,1)
        dist = (dist_per_step * mask.squeeze(-1)).sum(dim=-1) / valid_counts  # (bs, n, n)
        pairwise = -dist
    else:
        masked = candidates * valid_mask.unsqueeze(1)
        flat = masked.view(bs, n, -1)
        norm = flat.pow(2).sum(dim=2).clamp_min(1e-8).sqrt()
        pairwise = torch.matmul(flat, flat.transpose(1, 2)) / (norm.unsqueeze(2) * norm.unsqueeze(1) + 1e-8)

    idx = torch.triu_indices(n, n, offset=1, device=candidates.device)
    pairs = pairwise[:, idx[0], idx[1]]  # (bs, n*(n-1)/2)
    valid_rows = valid_mask.squeeze(-1).sum(dim=1) > 0
    pair_max = pairs.max(dim=1).values
    pair_min = pairs.min(dim=1).values
    return pair_max, pair_min, valid_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # deterministic sampling
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    # Load policy config first to infer chunk size, then build dataset and policy using dataset metadata.
    policy_cfg = _load_policy_config(args.policy_path)
    if args.n_action_steps is not None:
        policy_cfg.n_action_steps = args.n_action_steps
    chunk_size = getattr(policy_cfg, "n_action_steps", None)
    if chunk_size is None:
        raise ValueError("Policy config missing n_action_steps; specify --n-action-steps.")

    dataset_kwargs = dict(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        episodes=args.episodes,
        chunk_size=chunk_size,
        q_chunk_len=chunk_size,
        include_future_observation=False,
    )
    # Some LeRobot versions do not support `streaming`; only add if explicitly needed and supported.
    if args.streaming:
        logging.warning("`streaming` flag ignored because current LeRobotDataset does not accept it.")
    dataset = RewardAugmentedLeRobotDataset(**dataset_kwargs)
    policy, preprocessor, postprocessor = build_policy_bundle(
        policy_cfg=policy_cfg,
        policy_path=args.policy_path,
        device=str(device),
        ds_meta=dataset.meta,
    )

    # 默认输出路径时附带标识
    default_out = Path("outputs/simanalyse/eval_action_similarity.json")
    if args.output_json == default_out:
        suffix = []
        # suffix.append("aug" if args.use_data_augmentations else "noaug")
        # suffix.append("calql" if args.use_calql else "nocalql")
        args.output_json = default_out.with_name(f"{default_out.stem}_" + "_".join(suffix) + default_out.suffix)
    # 可选：若希望不带时间戳，保持原名
    stem_no_trailing_us = args.output_json.stem.rstrip("_")
    args.output_json = args.output_json.with_name(f"{stem_no_trailing_us}{args.output_json.suffix}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # random batches
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )

    iterator = iter(dataloader)
    try:
        raw_batch = next(iterator)
    except StopIteration:
        raise RuntimeError("Dataset is empty.")

    def ensure_action_key(batch: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in batch:
            action_key = next(
                (k for k, v in batch.items() if isinstance(v, torch.Tensor) and k.endswith("action")), None
            )
            if action_key:
                batch["action"] = batch[action_key]
            else:
                raise KeyError("No action tensor found in batch.")
        return batch

    batch = ensure_action_key(preprocessor(raw_batch))

    critic_trainer = build_critic(
        args.critic_state,
        policy,
        batch,
        str(device),
        args.action_samples,
        critic_q_agg_override=args.critic_q_agg,
    )
    critic_action_weights = getattr(critic_trainer.cfg, "action_distance_weights", None) if critic_trainer else None
    critic_q_agg_used = getattr(critic_trainer.cfg, "q_aggregation", None) if critic_trainer else None
    critic_loss_mode_used = getattr(critic_trainer.cfg, "critic_loss_mode", None) if critic_trainer else None

    reductions = ["flatten", "per_dim_mean"] if args.sim_reduction == "both" else [args.sim_reduction]

    base_stats = {
        "total_frames": 0,
        "total_samples": 0,
        # critic-related aggregates (reduction-agnostic)
        "q_mean_sum": 0.0,
        "q_max_sum": 0.0,
        "q_min_sum": 0.0,
        "gt_q_sum": 0.0,
        "gt_q_count": 0,
        "gt_sim": {},
        "q_frames": 0,
        # optional per-sample dump
        "per_sample": [],
    }

    stats_by_red: Dict[str, Dict[str, float]] = {}
    for red in reductions:
        stats_by_red[red] = {
            "sim_mean": 0.0,
            "sim_max_mean": 0.0,
            "sim_best_q_mean": 0.0,
            "sim_rank_sum": 0.0,
            "q_rank_sum": 0.0,
            "top1_hits": 0.0,
            "top3_hits": 0.0,
            "top4_hits": 0.0,
            "pairwise_max_sum": 0.0,
            "pairwise_min_sum": 0.0,
            "pairwise_frames": 0.0,
            "frames": 0.0,
            "gt_sim_sum": 0.0,
            "gt_sim_count": 0.0,
        }
    primary_reduction = reductions[0]

    def update_reduction_stats(
        red: str,
        sim_mat: torch.Tensor,
        best_q_sim: Optional[torch.Tensor],
        q_scores: Optional[torch.Tensor] = None,
        sim_rank_of_best_q: Optional[torch.Tensor] = None,
        q_rank_of_best_sim: Optional[torch.Tensor] = None,
        top1_hit: Optional[torch.Tensor] = None,
        top3_hit: Optional[torch.Tensor] = None,
        top4_hit: Optional[torch.Tensor] = None,
        gt_sim: Optional[torch.Tensor] = None,
        pairwise_max: Optional[torch.Tensor] = None,
        pairwise_min: Optional[torch.Tensor] = None,
    ) -> None:
        stats = stats_by_red[red]
        bs = sim_mat.shape[0]
        stats["frames"] += bs
        stats["sim_mean"] += float(sim_mat.mean().item()) * bs
        stats["sim_max_mean"] += float(sim_mat.max(dim=1).values.mean().item()) * bs
        if best_q_sim is not None:
            stats["sim_best_q_mean"] += float(best_q_sim.mean().item()) * bs
        if sim_rank_of_best_q is not None:
            stats["sim_rank_sum"] += float(sim_rank_of_best_q.sum().item())
        if q_rank_of_best_sim is not None:
            stats["q_rank_sum"] += float(q_rank_of_best_sim.sum().item())
        if top1_hit is not None:
            stats["top1_hits"] += float(top1_hit.sum().item())
        if top3_hit is not None:
            stats["top3_hits"] += float(top3_hit.sum().item())
        if top4_hit is not None:
            stats["top4_hits"] += float(top4_hit.sum().item())
        if gt_sim is not None:
            stats["gt_sim_sum"] += float(gt_sim.sum().item())
            stats["gt_sim_count"] += gt_sim.numel()
        if pairwise_max is not None and pairwise_min is not None:
            stats["pairwise_frames"] += bs
            stats["pairwise_max_sum"] += float(pairwise_max.sum().item())
            stats["pairwise_min_sum"] += float(pairwise_min.sum().item())

    # First batch processed; include in loop
    max_steps = args.limit_steps if args.limit_steps is not None else None
    step_idx = 0
    sample_id = 0

    action_weights_source = "cli/default"
    action_weights = args.action_weights
    if args.use_critic_weights and critic_action_weights:
        action_weights = list(critic_action_weights)
        action_weights_source = "critic_config"
    logging.info("Using action weights (%s): %s", action_weights_source, action_weights)

    while True:
        current_batch = batch if step_idx == 0 else None
        if current_batch is None:
            try:
                raw = next(iterator)
            except StopIteration:
                break
            current_batch = ensure_action_key(preprocessor(raw))

        gt_action = current_batch["action"]
        if gt_action.shape[1] < chunk_size:
            raise ValueError(f"GT action chunk shorter ({gt_action.shape[1]}) than chunk_size {chunk_size}.")
        pad_mask = _get_pad_mask(current_batch, chunk_size)
        gt_action = gt_action[:, :chunk_size].to(device)
        pad_mask_device = pad_mask.to(device) if pad_mask is not None else None
        with torch.no_grad():
            candidates, q, best_q = sample_candidates(
                policy=policy,
                batch=current_batch,
                critic_trainer=critic_trainer,
                action_samples=args.action_samples,
                device=str(device),
                chunk_size=chunk_size,
                use_current_critic=args.use_current_critic,
                forced_mask=pad_mask_device,
            )
            candidates = candidates.to(device)

            # Optional postprocess to env space (inverse normalization) before computing similarity.
            act_dim = candidates.shape[-1]
            if args.use_postprocess_sim:
                # use reshape (safe for non-contiguous tensors) for flatten + restore shapes
                gt_for_sim = postprocessor(gt_action.reshape(-1, act_dim)).reshape_as(gt_action).to(device)
                cand_for_sim = postprocessor(candidates.reshape(-1, act_dim)).reshape_as(candidates).to(device)
            else:
                gt_for_sim = gt_action
                cand_for_sim = candidates
            # 计算多种 reduction 的相似度
            sim_data: Dict[str, Dict[str, Any]] = {}
            for red in reductions:
                if red == "weighted":
                    sim_mat, valid_rows = weighted_l2_sim(
                        gt_for_sim, cand_for_sim, pad_mask_device, weights_list=action_weights
                    )
                else:
                    sim_mat, valid_rows = cosine_sim(gt_for_sim, cand_for_sim, pad_mask_device, reduction=red)
                pairwise_max, pairwise_min, pairwise_valid = pairwise_candidate_extremes(
                    cand_for_sim, pad_mask_device, reduction=red, weights_list=action_weights
                )
                pairwise_rows = valid_rows & pairwise_valid if pairwise_max is not None else valid_rows
                sim_data[red] = {
                    "sim_mat": sim_mat,
                    "valid_rows": valid_rows,
                    "pairwise_max": pairwise_max,
                    "pairwise_min": pairwise_min,
                    "pairwise_rows": pairwise_rows,
                }

            best_q_sim_by_red: Dict[str, torch.Tensor] = {}
            per_red_indices: Dict[str, Dict[str, torch.Tensor]] = {}
            gt_sim_by_red: Dict[str, torch.Tensor] = {}

            ref_key = primary_reduction if primary_reduction in sim_data else next(iter(sim_data.keys()))
            ref_rows = sim_data[ref_key]["pairwise_rows"]
            valid_count = int(ref_rows.sum().item())

            if q is not None and best_q is not None:
                q_scores = q.squeeze(-1)  # (bs, n)
                best_q_idx = torch.argmax(q_scores, dim=1)

                # critic score for ground-truth action (not used in ranking)
                if critic_trainer is not None:
                    critic_model = critic_trainer.critic if args.use_current_critic else critic_trainer.target_critic
                    gt_encoding = encode_policy_observations(policy, current_batch).to(device)
                    gt_action_mask = _get_pad_mask(current_batch, chunk_size)
                    if gt_action_mask is None:
                        gt_action_mask = torch.zeros(gt_action.shape[:2], dtype=torch.bool, device=device)
                    else:
                        gt_action_mask = gt_action_mask.to(device)
                    gt_q_vals = critic_model(gt_encoding, gt_action, action_mask=gt_action_mask)
                    gt_q_agg = critic_trainer._aggregate_q(*gt_q_vals).squeeze(-1)  # (bs,)
                else:
                    gt_q_agg = None

                # base q stats once using reference rows
                if valid_count > 0:
                    base_stats["total_frames"] += valid_count
                    base_stats["total_samples"] += valid_count * sim_data[ref_key]["sim_mat"].shape[1]
                    base_stats["q_frames"] += valid_count
                    base_stats["q_mean_sum"] += float(q_scores[ref_rows].mean(dim=1).sum().item())
                    base_stats["q_max_sum"] += float(q_scores[ref_rows].max(dim=1).values.sum().item())
                    base_stats["q_min_sum"] += float(q_scores[ref_rows].min(dim=1).values.sum().item())
                    if gt_q_agg is not None:
                        base_stats["gt_q_sum"] += float(gt_q_agg[ref_rows].sum().item())
                        base_stats["gt_q_count"] += gt_q_agg[ref_rows].numel()

                common_rows = ref_rows.clone()
                for red in reductions:
                    sim_mat = sim_data[red]["sim_mat"]
                    valid_rows = sim_data[red]["pairwise_rows"]
                    common_rows = common_rows & valid_rows
                    best_sim_idx = torch.argmax(sim_mat, dim=1)

                    sim_order = sim_mat.argsort(dim=1, descending=True)
                    q_order = q_scores.argsort(dim=1, descending=True)
                    best_q_idx_dev = best_q_idx.to(sim_order.device)
                    best_sim_idx_dev = best_sim_idx.to(q_order.device)
                    sim_rank_of_best_q = (sim_order == best_q_idx_dev.unsqueeze(1)).float().argmax(dim=1) + 1
                    q_rank_of_best_sim = (q_order == best_sim_idx_dev.unsqueeze(1)).float().argmax(dim=1) + 1

                    k3 = min(3, q_order.shape[1])
                    k4 = min(4, q_order.shape[1])
                    top3_hit = (q_order[:, :k3] == best_sim_idx_dev.unsqueeze(1)).any(dim=1).float()
                    top4_hit = (q_order[:, :k4] == best_sim_idx_dev.unsqueeze(1)).any(dim=1).float()
                    top1_hit = (best_q_idx_dev == best_sim_idx_dev.to(best_q_idx_dev.device)).float()

                    idx_device = sim_mat.device
                    batch_indices = torch.arange(sim_mat.shape[0], device=idx_device)
                    best_q_sim_red = sim_mat[batch_indices, best_q_idx.to(idx_device)]
                    best_q_sim_by_red[red] = best_q_sim_red

                    if red == "weighted":
                        gt_sim_red = weighted_l2_sim(
                            gt_for_sim, gt_for_sim.unsqueeze(1), pad_mask_device, weights_list=action_weights
                        )[0][:, 0]
                    else:
                        gt_sim_red = cosine_sim(gt_for_sim, gt_for_sim.unsqueeze(1), pad_mask_device, reduction=red)[0][
                            :, 0
                        ]
                    gt_sim_by_red[red] = gt_sim_red

                    if torch.any(valid_rows):
                        update_reduction_stats(
                            red,
                            sim_mat[valid_rows].cpu(),
                            best_q_sim_red[valid_rows].cpu(),
                            q_scores=q_scores[valid_rows].cpu(),
                            sim_rank_of_best_q=sim_rank_of_best_q[valid_rows].cpu(),
                            q_rank_of_best_sim=q_rank_of_best_sim[valid_rows].cpu(),
                            top1_hit=top1_hit[valid_rows].cpu(),
                            top3_hit=top3_hit[valid_rows].cpu(),
                            top4_hit=top4_hit[valid_rows].cpu(),
                            gt_sim=gt_sim_red[valid_rows].cpu(),
                            pairwise_max=sim_data[red]["pairwise_max"][valid_rows].cpu()
                            if sim_data[red]["pairwise_max"] is not None
                            else None,
                            pairwise_min=sim_data[red]["pairwise_min"][valid_rows].cpu()
                            if sim_data[red]["pairwise_min"] is not None
                            else None,
                        )

                    per_red_indices[red] = {
                        "best_q_idx": best_q_idx,
                        "best_sim_idx": best_sim_idx,
                        "sim_rank_of_best_q": sim_rank_of_best_q,
                        "q_rank_of_best_sim": q_rank_of_best_sim,
                        "top1_hit": top1_hit,
                        "top3_hit": top3_hit,
                        "top4_hit": top4_hit,
                        "pairwise_max": sim_data[red]["pairwise_max"],
                        "pairwise_min": sim_data[red]["pairwise_min"],
                    }

                # per-sample logging using intersection rows
                valid_indices = torch.nonzero(common_rows, as_tuple=False).flatten()
                if valid_indices.numel() > 0:
                    for i in valid_indices.tolist():
                        gt_q_val = float(gt_q_agg[i].item()) if gt_q_agg is not None else None
                        q_sim_pairs: Dict[str, Dict[str, float | None]] = {"gt": {"q": gt_q_val}}
                        for j in range(q_scores.shape[1]):
                            q_sim_pairs[str(j)] = {"q": float(q_scores[i, j].item())}
                        for red in reductions:
                            sim_mat = sim_data[red]["sim_mat"]
                            q_sim_pairs["gt"][f"{red}_sim"] = float(gt_sim_by_red[red][i].item())
                            for j in range(q_scores.shape[1]):
                                q_sim_pairs[str(j)][f"{red}_sim"] = float(sim_mat[i, j].item())

                        per_red_log = {}
                        for red in reductions:
                            per_red_log[red] = {
                                "best_q_idx": int(per_red_indices[red]["best_q_idx"][i]),
                                "best_sim_idx": int(per_red_indices[red]["best_sim_idx"][i]),
                                "sim_rank_of_best_q": int(per_red_indices[red]["sim_rank_of_best_q"][i]),
                                "q_rank_of_best_sim": int(per_red_indices[red]["q_rank_of_best_sim"][i]),
                                "top1_hit": bool(per_red_indices[red]["top1_hit"][i].item()),
                                "top3_hit": bool(per_red_indices[red]["top3_hit"][i].item()),
                                "top4_hit": bool(per_red_indices[red]["top4_hit"][i].item()),
                                "pairwise_sim_max": float(per_red_indices[red]["pairwise_max"][i].item())
                                if per_red_indices[red]["pairwise_max"] is not None
                                else None,
                                "pairwise_sim_min": float(per_red_indices[red]["pairwise_min"][i].item())
                                if per_red_indices[red]["pairwise_min"] is not None
                                else None,
                            }

                        base_stats["per_sample"].append(
                            {
                                "sample_id": sample_id,
                                "ground_truth_q": gt_q_val,
                                "q_sim_pairs": q_sim_pairs,
                                "per_reduction": per_red_log,
                            }
                        )
                        sample_id += 1

            else:
                # 无 critic 情况仅更新 sim/pairwise stats
                ref_key = primary_reduction if primary_reduction in sim_data else next(iter(sim_data.keys()))
                ref_rows = sim_data[ref_key]["pairwise_rows"]
                valid_count = int(ref_rows.sum().item())
                if valid_count > 0:
                    base_stats["total_frames"] += valid_count
                    base_stats["total_samples"] += valid_count * sim_data[ref_key]["sim_mat"].shape[1]
                for red in reductions:
                    sim_mat = sim_data[red]["sim_mat"]
                    valid_rows = sim_data[red]["pairwise_rows"]
                    if torch.any(valid_rows):
                        update_reduction_stats(
                            red,
                            sim_mat[valid_rows].cpu(),
                            best_q_sim=None,
                            pairwise_max=sim_data[red]["pairwise_max"][valid_rows].cpu()
                            if sim_data[red]["pairwise_max"] is not None
                            else None,
                            pairwise_min=sim_data[red]["pairwise_min"][valid_rows].cpu()
                            if sim_data[red]["pairwise_min"] is not None
                            else None,
                        )

        step_idx += 1
        if max_steps is not None and step_idx >= max_steps:
            break

    result = {
        "policy_path": str(args.policy_path),
        "critic_state": str(args.critic_state) if args.critic_state is not None else None,
        "critic_q_agg": critic_q_agg_used,
        "critic_loss_mode": critic_loss_mode_used,
        "dataset_root": str(args.dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "total_frames": base_stats["total_frames"],
        "samples_per_frame": args.action_samples,
        "postprocess_similarity": bool(args.use_postprocess_sim),
        "sim_reduction": args.sim_reduction,
        "action_weights": action_weights,
        "action_weights_source": action_weights_source,
    }
    if base_stats["q_frames"] > 0:
        denom = max(base_stats["q_frames"], 1)
        result.update(
            {
                "q_mean": base_stats["q_mean_sum"] / denom,
                "q_max": base_stats["q_max_sum"] / denom,
                "q_min": base_stats["q_min_sum"] / denom,
            }
        )
    if base_stats["gt_q_count"] > 0:
        result["ground_truth_q_mean"] = base_stats["gt_q_sum"] / max(base_stats["gt_q_count"], 1)

    # per-reduction aggregates
    for red in reductions:
        stats = stats_by_red[red]
        total = max(stats["frames"], 1.0)
        prefix = f"{red}_"
        result.update(
            {
                f"{prefix}sim_mean": stats["sim_mean"] / total,
                f"{prefix}sim_max_mean": stats["sim_max_mean"] / total,
            }
        )
        if stats["sim_best_q_mean"] > 0:
            result[f"{prefix}sim_best_q_mean"] = stats["sim_best_q_mean"] / total
        if stats["frames"] > 0:
            denom = max(stats["frames"], 1.0)
            result.update(
                {
                    f"{prefix}sim_rank_of_best_q": stats["sim_rank_sum"] / denom,
                    f"{prefix}q_rank_of_best_sim": stats["q_rank_sum"] / denom,
                    f"{prefix}top1_hit_rate": stats["top1_hits"] / denom,
                    f"{prefix}top3_hit_rate": stats["top3_hits"] / denom,
                    f"{prefix}top4_hit_rate": stats["top4_hits"] / denom,
                }
            )
        if stats["pairwise_frames"] > 0:
            denom_pw = max(stats["pairwise_frames"], 1.0)
            result[f"{prefix}pairwise_sim_max_mean"] = stats["pairwise_max_sum"] / denom_pw
            result[f"{prefix}pairwise_sim_min_mean"] = stats["pairwise_min_sum"] / denom_pw
        if stats["gt_sim_count"] > 0:
            result[f"{prefix}ground_truth_sim_mean"] = stats["gt_sim_sum"] / max(stats["gt_sim_count"], 1.0)

    # include per-sample details (may be large if limit_steps is big)
    result["per_sample"] = base_stats["per_sample"]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logging.info("Saved metrics to %s", args.output_json)
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
