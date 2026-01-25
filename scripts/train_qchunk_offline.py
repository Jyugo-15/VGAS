# train_with_critic_new_offline.py，上扩展至online训练
#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict, is_dataclass
from itertools import islice
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional
from types import MethodType
import json

import torch
from torchvision.utils import save_image
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import (
    ACTION,
    DONE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    REWARD,
    TRAINING_STATE_DIR,
    TRUNCATED,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from models.smolvla.modeling import make_att_2d_masks, resize_with_pad
from qchunk.critic_adapters import PolicyEmbeddings
from qchunk.critic_utils import aggregate_q, get_raw_state, get_tensor_from_batch, repeat_batch
from qchunk.qchunked_critic import QChunkedCritic
from qchunk.vgas_policy import VGASPolicy
from data.lerobot_reward_dataset import RewardAugmentedLeRobotDataset
from data.data_augmentations import vgps_augment, vgps_augment_vmap

REPO_ROOT = Path(__file__).resolve().parents[1]

CRITIC_STATE_FILE = "critic_state.pt"



@dataclass
class CriticConfig:
    enable: bool = True
    hidden_dims: tuple[int, int] = (512, 512)
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    discount: float = 0.99
    tau: float = 0.005
    grad_clip_norm: float = 10.0
    grad_clip_warmup: float | None = None
    action_samples: int = 8
    q_aggregation: str = "mean"  # {"mean", "min", "max"}
    use_calql: bool = False
    cql_m_actions: Optional[int] = None
    cql_alpha: float = 1.0
    cql_next_noise_std: float = 0.05
    cql_cur_noise_std: Optional[float] = None
    temperature: float = 1.0
    use_ood_reg: bool = False
    ood_alpha: float = 1.0
    ood_action_source: str = "erg"
    dist_penalty_beta: float = 0.5
    dist_clamp_max: float | None = None
    ood_include_current_actions: bool = True
    ood_include_random_actions: bool = False
    ood_include_next_actions: bool = False
    ood_noise_stds: tuple[float, ...] = (0.02,)
    use_ood_noise: bool = True
    use_ood_trunc: bool = True
    use_ood_mix: bool = False
    ood_mix_ratio: float = 0.5
    ood_mix_alpha_low: float = 0.3
    ood_mix_alpha_high: float = 0.7
    debug_mix_dist: bool = False
    use_pairwise_ood_loss: bool = False
    pairwise_ood_loss_weight: float = 1.0
    num_query_token: int = 16
    critic_type: str = "mlp"  # {"mlp", "q_chunk_former"}
    vqh_num_backbone_layers: int = 2
    vqh_hidden_dims: tuple[int, ...] = (512, 512)
    vqh_vlm_model_name: Optional[str] = None
    att_mode: str = "causal"
    head_type: str = "mlp"  # when using value_query_head
    head_num_layers: int = 2
    head_mlp_dims: tuple[int, ...] = (512, 512)
    num_q_heads: int = 1
    critic_loss_mode: str = "mse"
    value_head_num_layers: int = 2
    value_head_mlp_dims: tuple[int, ...] = (512, 512)
    value_head_vlm_model_name: Optional[str] = None
    lr_warmup_steps: int = 0
    lr_total_steps: Optional[int] = None
    lr_final: float = 0.0
    use_no_query_head: bool = False
    use_raw_state_fusion: bool = False
    raw_state_dim: int = 8
    q_chunk_len: int | None = None
    value_head_bias_init_enabled: bool = False
    value_head_bias_init_value: float = 0.0
    action_distance_weights: tuple[float, ...] | None = None
    mask_dropout_prob: float = 0.5
    use_dual_noise_ood: bool = False
    ood_warmup_steps: int = 0
    tau_warmup: float | None = None
    tau_warmup_steps: int = 0
    eval_ranking_freq: int = 0
    eval_ranking_batches: int = 8
    eval_ranking_action_samples: int = 8
    eval_ranking_batch_size: int | None = 32
    eval_ranking_start_step: int = 0
    eval_ranking_full_dataset_root: str | None = None


@dataclass
class TrainWithCriticPipelineConfig(TrainPipelineConfig):
    critic: CriticConfig = field(default_factory=CriticConfig)
    log_policy_to_wandb: bool = True
    log_code_to_wandb: bool = False
    code_artifact_dir: Path | None = None
    q_chunk_len: int | None = None
    critic_only: bool = False


def _build_reward_augmented_dataset(cfg: TrainWithCriticPipelineConfig):
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    if hasattr(cfg.policy, "chunk_size") and cfg.policy.chunk_size is not None:
        chunk_size = int(cfg.policy.chunk_size)
    elif hasattr(cfg, "qchunk"):
        chunk_size = int(getattr(cfg.qchunk, "chunk_size", 1))
    else:
        chunk_size = 1
    q_chunk_len = getattr(cfg, "q_chunk_len", None) or getattr(cfg.policy, "q_chunk_len", None)
    if q_chunk_len is None:
        q_chunk_len = getattr(cfg.policy, "n_action_steps", None)
    dataset = RewardAugmentedLeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        chunk_size=chunk_size,
        q_chunk_len=q_chunk_len,
        include_future_observation=True,
        max_action_dim=getattr(cfg.policy, "max_action_dim", None),
        video_backend=cfg.dataset.video_backend,
        discount=getattr(cfg.dataset, "discount", getattr(cfg.critic, "discount", 0.99)),
    )
    return dataset


def _merge_transition_keys(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Ensure reward/done tensors survive preprocessing by copying from raw batch if needed."""

    device = next((tensor.device for tensor in target.values() if isinstance(tensor, torch.Tensor)), None)
    for key in source:
        if key in source and key not in target:
            value = source[key]
            if isinstance(value, torch.Tensor) and device is not None:
                target[key] = value.to(device)
            else:
                target[key] = value


def _ensure_action_alias(batch: Dict[str, Any]) -> None:
    if "action" not in batch and ACTION in batch and isinstance(batch[ACTION], torch.Tensor):
        batch["action"] = batch[ACTION]


def _compute_spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_rank_idx = torch.argsort(torch.argsort(x))
    y_rank_idx = torch.argsort(torch.argsort(y))
    x_rank = x_rank_idx.float()
    y_rank = y_rank_idx.float()
    x_center = x_rank - x_rank.mean()
    y_center = y_rank - y_rank.mean()
    denom = (x_center.norm() * y_center.norm()).clamp(min=1e-8)
    corr = (x_center * y_center).sum() / denom
    return float(corr.item())


def _evaluate_critic_ranking(
    policy: PreTrainedPolicy,
    critic: QChunkedCritic,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    action_samples: int,
    action_weights: torch.Tensor,
    chunk_len: int,
    preprocessor: Any,
) -> Dict[str, float]:
    metrics: Dict[str, list[float]] = {
        "spearman_corr": [],
        "spearman_corr_candi": [],
        "top1_acc": [],
        "gt_rank": [],
        "effective_beta": [],
        "q_gap_gt_vs_avg": [],
        "sim_rank_of_best_q": [],
        "q_rank_of_best_sim": [],
        "sim_rank_of_best_q_candi": [],
        "q_rank_of_best_sim_candi": [],
        "top1_hit_rate_candi": [],
        "top3_hit_rate_candi": [],
        "top4_hit_rate_candi": [],
    }

    was_training_policy = policy.training
    was_training_critic = critic.critic.training
    policy.eval()
    critic.critic.eval()

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        with torch.no_grad():
            for batch in islice(dataloader, max_batches):
                raw_batch = batch
                _ = raw_batch.pop("next_observations", None)
                raw_snapshot = {k: v for k, v in raw_batch.items()}
                batch_proc = preprocessor(raw_batch)
                _merge_transition_keys(batch_proc, raw_batch)
                _ensure_action_alias(batch_proc)
                _ensure_reward_metadata(batch_proc, raw_snapshot)
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_proc.items()
                }
                if OBS_LANGUAGE_TOKENS not in batch_on_device:
                    logging.warning("Ranking eval skip batch: missing %s after preprocess", OBS_LANGUAGE_TOKENS)
                    continue
                encoding = critic.encode_policy(policy, batch_on_device).to(device)
                batch_size = batch_on_device["action"].shape[0]
                raw_state = get_raw_state(
                    batch_on_device,
                    batch_size,
                    getattr(critic.cfg, "raw_state_dim", critic._raw_state_dim),
                    device,
                )
                gt_actions = batch_on_device["action"]
                if gt_actions.shape[1] > chunk_len:
                    gt_actions = gt_actions[:, :chunk_len]
                gt_mask = get_tensor_from_batch(
                    batch_on_device,
                    ["actions_is_pad", "action_is_pad"],
                    default_shape=gt_actions.shape[:2],
                    device=device,
                )
                if gt_mask.shape[-1] > chunk_len:
                    gt_mask = gt_mask[..., :chunk_len]
                gt_mask = gt_mask.to(device).bool()

                expanded_batch = repeat_batch(batch_on_device, batch_size, action_samples)
                policy_candidates = policy.predict_action_chunk(expanded_batch)
                if policy_candidates.shape[1] < chunk_len:
                    raise ValueError(
                        f"Policy chunk ({policy_candidates.shape[1]}) shorter than q_chunk_len ({chunk_len})."
                    )
                if policy_candidates.shape[1] > chunk_len:
                    policy_candidates = policy_candidates[:, :chunk_len]
                policy_candidates = policy_candidates.view(batch_size, action_samples, chunk_len, -1)
                gt_expanded = gt_actions.unsqueeze(1)
                candidates = torch.cat([gt_expanded, policy_candidates], dim=1)

                diff = (candidates - gt_expanded) ** 2
                policy_mask = gt_mask.unsqueeze(1).expand(batch_size, action_samples, chunk_len)
                combined_mask = torch.cat([gt_mask.unsqueeze(1), policy_mask], dim=1)
                valid_mask = (~combined_mask).unsqueeze(-1)
                weighted = diff * action_weights
                weighted = weighted * valid_mask
                valid_count = valid_mask.sum(dim=(-1, -2)).clamp(min=1.0)
                dists = weighted.sum(dim=(-1, -2)) / valid_count

                batch_size, num_candidates, steps, act_dim = candidates.shape
                flat_candidates = candidates.view(-1, steps, act_dim)
                flat_mask = combined_mask.view(-1, steps)
                rep_encoding = encoding.repeat(num_candidates)
                rep_raw_state = raw_state.repeat_interleave(num_candidates, dim=0)

                q_vals = critic.critic(
                    rep_encoding,
                    flat_candidates,
                    action_mask=flat_mask,
                    raw_state=rep_raw_state,
                )
                q_vals = aggregate_q(q_vals, getattr(critic.cfg, "q_aggregation", "mean")).view(
                    batch_size, num_candidates
                )

                d_cpu = dists.cpu()
                q_cpu = q_vals.cpu()
                for b in range(batch_size):
                    d = d_cpu[b]
                    q = q_cpu[b]
                    corr = _compute_spearman_corr(-d, q)
                    metrics["spearman_corr"].append(corr)
                    if q.shape[0] > 1:
                        corr_candi = _compute_spearman_corr(-d[1:], q[1:])
                        metrics["spearman_corr_candi"].append(corr_candi)
                    best_dist_idx = int(torch.argmin(d).item())
                    best_q_idx = int(torch.argmax(q).item())
                    dist_order = torch.argsort(d, descending=False)
                    q_order = torch.argsort(q, descending=True)
                    sim_rank_best_q = int((dist_order == best_q_idx).nonzero(as_tuple=False).item()) + 1
                    q_rank_best_sim = int((q_order == best_dist_idx).nonzero(as_tuple=False).item()) + 1
                    metrics["sim_rank_of_best_q"].append(float(sim_rank_best_q))
                    metrics["q_rank_of_best_sim"].append(float(q_rank_best_sim))
                    metrics["top1_acc"].append(1.0 if best_dist_idx == best_q_idx else 0.0)
                    if q.shape[0] > 1:
                        d_candi = d[1:]
                        q_candi = q[1:]
                        best_candi_idx = int(torch.argmin(d_candi).item())
                        q_order_candi = torch.argsort(q_candi, descending=True)
                        dist_order_candi = torch.argsort(d_candi, descending=False)
                        best_q_candi_idx = int(torch.argmax(q_candi).item())
                        sim_rank_best_q_c = int((dist_order_candi == best_q_candi_idx).nonzero(as_tuple=False).item()) + 1
                        q_rank_best_sim_c = int((q_order_candi == best_candi_idx).nonzero(as_tuple=False).item()) + 1
                        metrics["sim_rank_of_best_q_candi"].append(float(sim_rank_best_q_c))
                        metrics["q_rank_of_best_sim_candi"].append(float(q_rank_best_sim_c))
                        metrics["top1_hit_rate_candi"].append(
                            1.0 if best_candi_idx == int(q_order_candi[0].item()) else 0.0
                        )
                        top3_hit_c = 1.0 if best_candi_idx in q_order_candi[:3].tolist() else 0.0
                        top4_hit_c = 1.0 if best_candi_idx in q_order_candi[:4].tolist() else 0.0
                        metrics["top3_hit_rate_candi"].append(top3_hit_c)
                        metrics["top4_hit_rate_candi"].append(top4_hit_c)
                    gt_rank = int((q_order == 0).nonzero(as_tuple=False).item())
                    metrics["gt_rank"].append(gt_rank)
                    q_gt = float(q[0].item())
                    q_others = float(q[1:].mean().item()) if q.shape[0] > 1 else q_gt
                    dist_others = float(d[1:].mean().item()) if d.shape[0] > 1 else 0.0
                    if dist_others > 1e-6:
                        slope = (q_gt - q_others) / dist_others
                        metrics["effective_beta"].append(slope)
                    metrics["q_gap_gt_vs_avg"].append(q_gt - q_others)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if was_training_policy:
            policy.train()
        if was_training_critic:
            critic.critic.train()

    summary = {
        k: float(torch.tensor(v, dtype=torch.float32).mean().item()) if len(v) > 0 else 0.0
        for k, v in metrics.items()
    }
    return summary

def _ensure_reward_metadata(batch: Dict[str, Any], raw_snapshot: Dict[str, Any]) -> None:
    rewards = batch.get("rewards")
    if rewards is None and REWARD in batch and isinstance(batch[REWARD], torch.Tensor):
        rewards = batch[REWARD]
        batch["rewards"] = rewards
    if rewards is None and REWARD in raw_snapshot and isinstance(raw_snapshot[REWARD], torch.Tensor):
        rewards = raw_snapshot[REWARD]
        batch["rewards"] = rewards

    if "reward_is_pad" in batch:
        return

    pad_tensor = None
    for key in ("reward_is_pad", "rewards_is_pad", "reward_pad", "actions_is_pad"):
        candidate = batch.get(key)
        if isinstance(candidate, torch.Tensor):
            pad_tensor = candidate
            break
        candidate = raw_snapshot.get(key)
        if isinstance(candidate, torch.Tensor):
            pad_tensor = candidate
            break

    if pad_tensor is None and isinstance(rewards, torch.Tensor):
        pad_tensor = torch.zeros_like(rewards, dtype=torch.bool)

    if isinstance(pad_tensor, torch.Tensor):
        target_device = rewards.device if isinstance(rewards, torch.Tensor) else pad_tensor.device
        pad_tensor = pad_tensor.to(device=target_device, dtype=torch.bool)
        batch["reward_is_pad"] = pad_tensor


def _propagate_future_pad(processed_future: Optional[Dict[str, Any]], original_future: Optional[Dict[str, Any]]) -> None:
    if not processed_future or not original_future:
        return
    # Preserve padding/meta signals that may be stripped by preprocessing.
    for key in ("next_observation_is_pad", "next_obs_valid_chunk_len"):
        if key in original_future and key not in processed_future:
            processed_future[key] = original_future[key]


def _build_ranking_eval_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(torch.seed())
    sampler = torch.utils.data.RandomSampler(dataset, generator=gen)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


def my_prepare_images(self, batch):
    """Custom image preprocessing for experiments; replace policy.prepare_images with this when needed."""

    # viz_dir = Path("")
    # viz_dir.mkdir(parents=True, exist_ok=True)

    # def _save_debug(tensor: torch.Tensor, name: str) -> None:
    #     save_image(tensor.detach().cpu().clamp(0.0, 1.0), viz_dir / name)
    #     # Also save per-channel views for debugging.
    #     if tensor.ndim == 3 and tensor.shape[0] >= 3:
    #         colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    #         stem = name.rsplit(".", 1)[0]
    #         for c in range(3):
    #             ch = tensor[c : c + 1].detach().cpu()
    #             color = colors[c].view(3, 1, 1)
    #             colored = ch * color  # map channel to its original color
    #             save_image(colored, viz_dir / f"{stem}_c{c}.png")

    present_img_keys = [key for key in self.config.image_features if key in batch]
    missing_img_keys = [key for key in self.config.image_features if key not in batch]

    if len(present_img_keys) == 0:
        raise ValueError(
            f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
        )

    images: list[torch.Tensor] = []
    img_masks: list[torch.Tensor] = []
    for key in present_img_keys:
        ########################################################################################
        img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
        # _save_debug(img[0], f"{key}_raw.png")

        if self.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
        # _save_debug(img[0], f"{key}_resized.png")

        # Keep augment resize target equal to current image size (e.g., 512x512).
        
        img = vgps_augment_vmap(img, image_size=img.shape[-2:])
        # _save_debug(img[0], f"{key}_aug.png")
        ########################################################################################
        # # Example tweak: add mild noise during training to reduce overfitting.
        # if self.training:
        #     noise_std = 0.01
        #     img = torch.clamp(img + torch.randn_like(img) * noise_std, 0.0, 1.0)

        img = img * 2.0 - 1.0
        # _save_debug(img[0] , f"{key}_norm.png")

        bsize = img.shape[0]
        device = img.device
        if f"{key}_padding_mask" in batch:
            mask = batch[f"{key}_padding_mask"].bool()
        else:
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
        images.append(img)
        img_masks.append(mask)

    for num_empty_cameras in range(len(missing_img_keys)):
        if num_empty_cameras >= self.config.empty_cameras:
            break
        img = torch.ones_like(images[0]) * -1
        mask = torch.zeros_like(img_masks[0])
        images.append(img)
        img_masks.append(mask)
    return images, img_masks


# def encode_policy_observations_test( # 封装数据增强以及是否使用vlm backbone提取特征
#     policy: PreTrainedPolicy,
#     batch: Dict[str, torch.Tensor],
#     use_data_augmentations: bool = False,
#     use_vlm_backbone_encode: bool = True,
# ) -> PolicyEmbeddings:
#     """Encode observations via the SmolVLA backbone, detached from gradients."""
#     # 增加数据增强 目的是训练数据偏少的情况下容易过拟合
#     training = policy.training
#     policy.eval()
#     processed_batch = policy._prepare_batch({k: v for k, v in batch.items()})
#     old = policy.prepare_images
#     try:
#         if use_data_augmentations:
#             policy.prepare_images = MethodType(my_prepare_images, policy)
#         images, img_masks = policy.prepare_images(processed_batch)
#     finally:
#         policy.prepare_images = old

#     state = policy.prepare_state(processed_batch)
#     lang_tokens = processed_batch[f"{OBS_LANGUAGE_TOKENS}"]
#     lang_masks = processed_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

#     prefix_embs, pad_masks, att_masks = policy.model.embed_prefix(
#         images, img_masks, lang_tokens, lang_masks, state=state
#     )
#     att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
#     position_ids = torch.cumsum(pad_masks, dim=1) - 1
#     # ******
#     if use_vlm_backbone_encode:
#         outputs_embeds, _ = policy.model.vlm_with_expert.forward(
#             attention_mask=att_2d_masks,
#             position_ids=position_ids,
#             past_key_values=None,
#             inputs_embeds=[prefix_embs, None],
#             use_cache=False,
#             fill_kv_cache=True,
#         )

#         prefix_outputs = outputs_embeds[0].to(torch.float32)
#     else:
#         prefix_outputs = prefix_embs
#     pad_mask_bool = pad_masks.bool()
#     att_mask_bool = att_masks.bool()

#     def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         mask = mask.unsqueeze(-1).to(tensor.dtype)
#         summed = (tensor * mask).sum(dim=1)
#         denom = mask.sum(dim=1).clamp_min(1.0)
#         return summed / denom

#     content_mask = (~att_mask_bool) & pad_mask_bool
#     state_mask = att_mask_bool & pad_mask_bool

#     img_lang_emb = masked_mean(prefix_outputs, content_mask)
#     state_emb = masked_mean(prefix_outputs, state_mask)
#     embedding = torch.cat([img_lang_emb, state_emb], dim=-1)
#     policy.train(training)
#     return PolicyEmbeddings(
#         pooled=embedding.detach(),
#         prefix_outs=prefix_outputs.detach(),
#         pad_masks=pad_masks.detach(),
#         att_masks=att_masks.detach(),
#     )


def encode_policy_observations(policy: PreTrainedPolicy, batch: Dict[str, torch.Tensor]) -> PolicyEmbeddings:
    """Encode observations via the SmolVLA backbone, detached from gradients."""

    training = policy.training
    policy.eval()
    processed_batch = policy._prepare_batch({k: v for k, v in batch.items()})
    images, img_masks = policy.prepare_images(processed_batch)
    state = policy.prepare_state(processed_batch)
    lang_tokens = processed_batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = processed_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    prefix_embs, pad_masks, att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    # ******
    outputs_embeds, _ = policy.model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        fill_kv_cache=True,
    )

    prefix_outputs = outputs_embeds[0].to(torch.float32)
    pad_mask_bool = pad_masks.bool()
    att_mask_bool = att_masks.bool()

    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(tensor.dtype)
        summed = (tensor * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    content_mask = (~att_mask_bool) & pad_mask_bool
    state_mask = att_mask_bool & pad_mask_bool

    img_lang_emb = masked_mean(prefix_outputs, content_mask)
    state_emb = masked_mean(prefix_outputs, state_mask)
    embedding = torch.cat([img_lang_emb, state_emb], dim=-1)
    policy.train(training)
    return PolicyEmbeddings(
        pooled=embedding.detach(),
        prefix_outs=prefix_outputs.detach(),
        pad_masks=pad_masks.detach(),
        att_masks=att_masks.detach(),
    )


def _sanitize_for_wandb(payload: dict) -> dict:
    """Convert tensors or unsupported types into plain python scalars for wandb logging."""

    sanitized: dict[str, float | int | str] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.numel() == 1:
                sanitized[key] = float(tensor.item())
            else:
                sanitized[key] = float(tensor.mean().item())
        elif isinstance(value, (float, int, str)):
            sanitized[key] = value
        elif value is None:
            continue
    return sanitized


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. It also handles mixed-precision training via a GradScaler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        grad_scaler: The GradScaler for automatic mixed-precision training.
        lr_scheduler: An optional learning rate scheduler.
        use_amp: A boolean indicating whether to use automatic mixed precision.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainWithCriticPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
    """
    cfg.validate()
    cfg_dict = cfg.to_dict()
    logging.info(pformat(cfg_dict))
    logging.info(pformat(cfg.to_dict()))
    local_config_path = Path(
        ""
    )
    import json
    with open(local_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)
    critic_only = getattr(cfg, "critic_only", False)

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
        if cfg.log_code_to_wandb:
            code_dir = cfg.code_artifact_dir or REPO_ROOT
            wandb_logger.log_code(code_dir)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    try:
        dataset = _build_reward_augmented_dataset(cfg)
        logging.info("Using RewardAugmentedLeRobotDataset for training.")
    except Exception as exc:  # pragma: no cover
        logging.warning("RewardAugmented dataset build failed (%s); falling back to default factory.", exc)
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        cfg.eval.batch_size = 3
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    if critic_only:
        for p in policy.parameters():
            p.requires_grad_(False)
        policy.eval()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if not critic_only:
        logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    else:
        optimizer = None
        lr_scheduler = None
        grad_scaler = None

    step = 0  # number of policy updates (forward + backward + optim)

    critic_state_resume_dir: Optional[Path] = None
    if cfg.resume:
        if critic_only:
            critic_state_resume_dir = Path(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None
        else:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
            if cfg.checkpoint_path is not None:
                critic_state_resume_dir = Path(cfg.checkpoint_path)
        # 恢复 critic_only 模式下的 step 计数（若存在），否则尝试从目录名推断
        if critic_only and cfg.checkpoint_path is not None:
            resume_step = load_critic_step(Path(cfg.checkpoint_path))
            if resume_step is None:
                try:
                    resume_step = int(Path(cfg.checkpoint_path).name)
                except ValueError:
                    resume_step = None
            if resume_step is not None:
                step = resume_step

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info("%s %s", colored("Output dir:", "yellow", attrs=["bold"]), cfg.output_dir)
    if cfg.env is not None:
        logging.info("cfg.env.task=%s", cfg.env.task)
    logging.info("cfg.steps=%s (%s)", cfg.steps, format_big_number(cfg.steps))
    logging.info(
        "dataset.num_frames=%s (%s)",
        dataset.num_frames,
        format_big_number(dataset.num_frames),
    )
    logging.info("dataset.num_episodes=%s", dataset.num_episodes)
    logging.info(
        "num_learnable_params=%s (%s)",
        num_learnable_params,
        format_big_number(num_learnable_params),
    )
    logging.info(
        "num_total_params=%s (%s)",
        num_total_params,
        format_big_number(num_total_params),
    )

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    if cfg.critic.enable:
        train_metrics["critic_loss"] = AverageMeter("critic_loss", ":.3f")
        train_metrics["critic_q"] = AverageMeter("crit_q", ":.3f")
        train_metrics["critic_lr"] = AverageMeter("crit_lr", ":0.1e")
        train_metrics["critic_update_s"] = AverageMeter("crit_updt_s", ":.3f")
        train_metrics["td_loss"] = AverageMeter("td_loss", ":.3f")
        train_metrics["td_error_mean"] = AverageMeter("td_err", ":.3f")
        num_q_heads = getattr(cfg.critic, "num_q_heads", 1)
        for idx in range(num_q_heads):
            name = f"critic_q_head{idx + 1}"
            train_metrics[name] = AverageMeter(name, ":.3f")
        for idx in range(num_q_heads):
            name = f"target_q_head{idx + 1}"
            train_metrics[name] = AverageMeter(name, ":.3f")
        if getattr(cfg.critic, "use_calql", False):
            calql_metric_names = [
                ("calql_loss", "calql_loss"),
                ("calql_ood_q", "calql_ood_q"),
                ("calql_q_batch", "calql_q_batch"),
                ("calql_ood_q_mean", "calql_ood_mean"),
                ("calql_rand_q_mean", "calql_rand_q"),
                ("calql_next_q_mean", "calql_next_q"),
                ("calql_cur_q_mean", "calql_cur_q"),
                ("calql_bound_rate_next", "calql_bound_rate_next"),
                ("calql_bound_rate_cur", "calql_bound_rate_cur"),
            ]
            for key, short in calql_metric_names:
                train_metrics[key] = AverageMeter(short, ":.3f")
        ood_metric_names = [
            ("ood_loss", "ood_loss"),
            ("ood_q_mean", "ood_q"),
            ("ood_dist_mean", "ood_dist"),
            ("ood_target_mean", "ood_tgt"),
        ]
        for key, short in ood_metric_names:
            train_metrics[key] = AverageMeter(short, ":.3f")
        # Additional OOD diagnostics
        extra_ood_metrics = [
            ("win_rate/gt_vs_policy", "win_pol"),
            ("win_rate/gt_vs_trunc", "win_trunc"),
            ("gap/gt_vs_policy", "gap_pol"),
            ("gap/gt_vs_trunc", "gap_trunc"),
            ("align/top1_policy_best_sim", "align_top1"),
            ("align/rank_policy_best_sim", "align_rank"),
            ("stability/head_diff_ood", "head_diff"),
            ("debug/raw_penalty_score", "raw_pen"),
            ("act_norm/policy_candidates", "act_pol"),
            ("act_norm/gt_clean", "act_gt"),
        ]
        for key, short in extra_ood_metrics:
            train_metrics[key] = AverageMeter(short, ":.3f")

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    critic: Optional[QChunkedCritic] = None
    vgas_policy: Optional[VGASPolicy] = None
    ranking_loader: Optional[DataLoader] = None
    ranking_full_loader: Optional[DataLoader] = None
    ranking_full_dataset = None
    ######### offline step
    for _ in range(step, cfg.steps):
        # if _ % 10 == 0:
        #     print("current step is %s" % _)
        start_time = time.perf_counter()
        raw_batch = next(dl_iter)
        next_observations = raw_batch.pop("next_observations", None)
        raw_batch_snapshot = {key: value for key, value in raw_batch.items()}
        batch = preprocessor(raw_batch)
        _merge_transition_keys(batch, raw_batch)
        _ensure_action_alias(batch)
        _ensure_reward_metadata(batch, raw_batch_snapshot)
        if next_observations is not None:
            future_sample = {key: value for key, value in raw_batch_snapshot.items()}
            for key, value in next_observations.items():
                future_sample[key] = value
            processed_future = preprocessor(future_sample)
            _propagate_future_pad(processed_future, next_observations)
            batch["next_observations"] = processed_future
        train_tracker.dataloading_s = time.perf_counter() - start_time

        
        if cfg.critic.enable and critic is None:
            critic = QChunkedCritic.build(
                policy_path=Path(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else Path(cfg.output_dir),
                policy_cfg=cfg.policy,
                critic_cfg=cfg.critic,
                sample_batch=batch,
                ds_meta=dataset.meta,
                device=device,
                encoder_fn=lambda p, b: encode_policy_observations(
                    p,
                    b,
                    use_data_augmentations=getattr(cfg, "use_data_augmentations", False),
                    use_vlm_backbone_encode=getattr(cfg, "use_vlm_backbone_encode", True),
                ),
                actor=policy,
            )
            vgas_policy = VGASPolicy(actor=policy, critic=critic)
            if critic_state_resume_dir is not None:
                print("reusing Critic")
                load_critic_state(critic, critic_state_resume_dir)
                critic_state_resume_dir = None

        output_dict: Dict[str, Any] = {}
        if not critic_only:
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )
        critic_metrics = None
        if vgas_policy is not None:
            critic_start = time.perf_counter()
            warmup_steps = getattr(cfg.critic, "ood_warmup_steps", 0)
            critic_metrics = vgas_policy.update_critic(
                batch,
                current_step=step,
                ood_warmup_steps=warmup_steps,
            )
            critic_metrics["critic_update_s"] = time.perf_counter() - critic_start
            for metric_name, metric_value in critic_metrics.items():
                if metric_name in train_tracker.metrics:
                    setattr(train_tracker, metric_name, metric_value)
            if not getattr(cfg.critic, "use_calql", False):
                # populate CalQL metrics with zeros for consistent logging when CalQL is disabled
                calql_keys = [
                    "calql_loss",
                    "calql_ood_q",
                    "calql_q_batch",
                    "calql_ood_q_mean",
                    "calql_rand_q_mean",
                    "calql_next_q_mean",
                    "calql_cur_q_mean",
                    "calql_bound_rate_next",
                    "calql_bound_rate_cur",
                ]
                for key in calql_keys:
                    if key in train_tracker.metrics:
                        setattr(train_tracker, key, 0.0)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if critic_metrics:
                    wandb_log_dict.update(critic_metrics)
                wandb_logger.log_dict(_sanitize_for_wandb(wandb_log_dict), step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info("Checkpoint policy after step %s", step)
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            if not critic_only:
                save_checkpoint(
                    checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor
                )
                if wandb_logger and cfg.log_policy_to_wandb:
                    wandb_logger.log_policy(checkpoint_dir)
            save_critic_state(critic, checkpoint_dir)
            save_critic_step(step, checkpoint_dir)
            update_last_checkpoint(checkpoint_dir)

        eval_freq = getattr(cfg.critic, "eval_ranking_freq", 0)
        eval_start = max(0, int(getattr(cfg.critic, "eval_ranking_start_step", 0)))
        if eval_freq > 0 and step > eval_start and step % eval_freq == 0 and critic is not None:
            batch_size_eval = getattr(cfg.critic, "eval_ranking_batch_size", None) or cfg.batch_size
            if ranking_loader is None:
                ranking_loader = _build_ranking_eval_loader(
                    dataset,
                    batch_size_eval,
                    num_workers=cfg.num_workers,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=2,
                )
            action_samples_eval = getattr(cfg.critic, "eval_ranking_action_samples", 8)
            max_batches_eval = getattr(cfg.critic, "eval_ranking_batches", 4)
            weights = cfg.critic.action_distance_weights or (5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0)
            action_weights = torch.tensor(weights, device=device, dtype=torch.float32).view(1, 1, 1, -1)
            ranking_metrics = _evaluate_critic_ranking(
                policy,
                critic,
                ranking_loader,
                device,
                max_batches=max_batches_eval,
                action_samples=action_samples_eval,
                action_weights=action_weights,
                chunk_len=critic.q_chunk_len,
                preprocessor=preprocessor,
            )
            rank_time = time.perf_counter() - start_time
            metrics_to_log = {f"rank_eval/train/{k}": v for k, v in ranking_metrics.items()}
            metrics_to_log["rank_eval/train/time_s"] = rank_time

            full_root = getattr(cfg.critic, "eval_ranking_full_dataset_root", None)
            if full_root:
                if ranking_full_loader is None:
                    tmp_cfg = copy.deepcopy(cfg)
                    tmp_cfg.dataset = copy.deepcopy(cfg.dataset)
                    tmp_cfg.dataset.root = str(full_root)
                    tmp_cfg.dataset.include_future_observation = False
                    try:
                        ranking_full_dataset = _build_reward_augmented_dataset(tmp_cfg)
                    except Exception as exc:
                        logging.warning("Full-data ranking dataset build failed (%s); skipping full eval.", exc)
                        ranking_full_dataset = None
                    if ranking_full_dataset is not None:
                        ranking_full_loader = _build_ranking_eval_loader(
                            ranking_full_dataset,
                            batch_size_eval,
                            num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda",
                            prefetch_factor=2,
                        )
                if ranking_full_loader is not None:
                    start_full = time.perf_counter()
                    full_metrics = _evaluate_critic_ranking(
                        policy,
                        critic,
                        ranking_full_loader,
                        device,
                        max_batches=max_batches_eval,
                        action_samples=action_samples_eval,
                        action_weights=action_weights,
                        chunk_len=critic.q_chunk_len,
                        preprocessor=preprocessor,
                    )
                    rank_full_time = time.perf_counter() - start_full
                    metrics_to_log.update({f"rank_eval/full/{k}": v for k, v in full_metrics.items()})
                    metrics_to_log["rank_eval/full/time_s"] = rank_full_time

            if wandb_logger:
                wandb_logger.log_dict(metrics_to_log, step)
            logging.info(
                "RankEval step %s | train spearman=%.4f top1=%.4f gt_rank=%.2f eff_beta=%.4f q_gap=%.4f time=%.2fs%s",
                step,
                ranking_metrics["spearman_corr"],
                ranking_metrics["top1_acc"],
                ranking_metrics["gt_rank"],
                ranking_metrics["effective_beta"],
                ranking_metrics["q_gap_gt_vs_avg"],
                rank_time,
                (
                    f" | full spearman={metrics_to_log.get('rank_eval/full/spearman_corr', 0.0):.4f}"
                    f" top1={metrics_to_log.get('rank_eval/full/top1_acc', 0.0):.4f}"
                    f" gt_rank={metrics_to_log.get('rank_eval/full/gt_rank', 0.0):.2f}"
                    f" eff_beta={metrics_to_log.get('rank_eval/full/effective_beta', 0.0):.4f}"
                    f" q_gap={metrics_to_log.get('rank_eval/full/q_gap_gt_vs_avg', 0.0):.4f}"
                    f" time={metrics_to_log.get('rank_eval/full/time_s', 0.0):.2f}s"
                )
            )

    
def _critic_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / TRAINING_STATE_DIR / CRITIC_STATE_FILE


def save_critic_state(critic: Optional[QChunkedCritic], checkpoint_dir: Path) -> None:
    if critic is None:
        return
    critic_path = checkpoint_dir / "critic_pretrained_model"
    critic_path.mkdir(parents=True, exist_ok=True)
    last_state = critic_path / "last.ckpt"
    payload = {
        "state_dict": critic.state_dict(),
        "meta": {
            "chunk_size": getattr(critic, "chunk_size", None),
            "action_step_dim": getattr(critic, "action_step_dim", None),
        },
    }
    cfg = getattr(critic, "cfg", None)
    if cfg is not None and is_dataclass(cfg):
        payload["critic_config"] = asdict(cfg)
    torch.save(payload, last_state)


def load_critic_state(critic: Optional[QChunkedCritic], checkpoint_dir: Path) -> None:
    if critic is None:
        return
    critic_path = checkpoint_dir / "critic_pretrained_model" / "last.ckpt"
    if critic_path.exists():
        payload = torch.load(critic_path, map_location=critic.device)
        state = payload.get("state_dict", payload)
        critic.load_state_dict(state)


def _critic_step_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "critic_pretrained_model" / "training_step.json"


def save_critic_step(step: int, checkpoint_dir: Path) -> None:
    step_path = _critic_step_path(checkpoint_dir)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    with step_path.open("w", encoding="utf-8") as f:
        json.dump({"step": step}, f)


def load_critic_step(checkpoint_dir: Path) -> Optional[int]:
    step_path = _critic_step_path(checkpoint_dir)
    if not step_path.exists():
        return None
    try:
        with step_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        step_val = data.get("step", None)
        return int(step_val) if step_val is not None else None
    except Exception:
        return None


def train_from_config(cfg: TrainWithCriticPipelineConfig):
    return train(cfg)


def main():
    
    init_logging()
    train()

def set_cuda():
    import os
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available =", torch.cuda.is_available())
    print("device_count =", torch.cuda.device_count())

    if torch.cuda.is_available():
        cur_idx = torch.cuda.current_device()        # 进程内索引（0..N-1）
        print("current_device index =", cur_idx)
        print("current_device name  =", torch.cuda.get_device_name(cur_idx))

if __name__ == "__main__":
    main()
