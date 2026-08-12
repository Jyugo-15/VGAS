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
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional
from types import MethodType
import json

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    REWARD,
    TRAINING_STATE_DIR,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
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
from models.smolvla.modeling_smolvla import make_att_2d_masks, resize_with_pad
from qchunk.critic_adapters import PolicyEmbeddings
from qchunk.distill_helpers import (
    _build_distill_actor_batch,
    _build_teacher_policy,
    _sample_policy_actions_single,
    _sanitize_for_wandb,
    _select_best_teacher_actions,
    _select_local_optimized_student_actions,
)
from qchunk.qchunked_critic import QChunkedCritic
from qchunk.vgas_policy import VGASPolicy
from data.lerobot_reward_dataset import RewardAugmentedLeRobotDataset
from data.data_augmentations import vgps_augment_vmap
from utils.local_policy import make_local_smolvla_policy

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
    critic_updates_per_step: int = 1
    action_samples: int = 4
    backup_action_samples: int | None = None  # preferred alias for Expected-Max backup samples
    q_aggregation: str = "mean"  # {"mean", "min", "max"}
    use_calql: bool = False
    ood_m_actions: Optional[int] = None
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
    loss_anchor_weight: float = 1.0
    loss_rank_weight: float = 1.0
    num_query_token: int = 16
    critic_type: str = "mlp"  # {"mlp", "q_chunk_former"}
    qformer_num_backbone_layers: int = 2
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
    scheduler_step_mode: str = "critic_update"
    use_no_query_head: bool = False
    use_raw_state_fusion: bool = False
    use_vlm_backbone_encode: bool = True
    use_critic_state_encoder: bool = False
    use_independent_critic_encoder: bool = False
    critic_vision_freeze_layers: int = 8
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


@dataclass
class DistillConfig:
    enable: bool = False
    teacher_pretrained_path: str | None = None
    teacher_num_steps: int | None = None
    student_num_steps: int | None = None
    mask_padded_action_loss: bool = True
    phase1_teacher_action_samples: int = 1
    action_samples: int | None = None  # deprecated alias, kept for backward compatibility
    freeze_teacher: bool = True
    phase1_steps: int = 5000
    phase1_train_student: bool = False
    phase1_use_student_as_source: bool = False
    phase1_critic_updates_per_step: int | None = None
    phase2_global_samples: int = 8
    local_samples: int = 4
    phase2_include_dataset_action_in_global_pool: bool = False
    phase2_local_opt_steps: int = 5
    phase2_local_opt_lr: float = 3e-4
    phase2_local_grad_normalize: bool = True
    phase2_local_adv_weight: bool = False
    phase2_local_adv_eps: float = 1e-6
    phase2_disable_critic_best_of_n: bool = False
    phase2_use_teacher_anchor: bool = True
    phase2_loss_rank_weight: float | None = None
    lambda_w2: float = 0.2
    lambda_opt: float = 1.0


@dataclass
class TrainWithCriticPipelineConfig(TrainPipelineConfig):
    critic: CriticConfig = field(default_factory=CriticConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    log_policy_to_wandb: bool = True
    log_code_to_wandb: bool = False
    code_artifact_dir: Path | None = None
    q_chunk_len: int | None = None
    critic_init_checkpoint: Path | None = None


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
        allow_missing_reward=getattr(cfg.dataset, "allow_missing_reward", False),
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

def encode_policy_observations(policy: PreTrainedPolicy, batch: Dict[str, torch.Tensor], use_vlm_backbone_encode: bool = True) -> PolicyEmbeddings:
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
    if use_vlm_backbone_encode:
        outputs_embeds, _ = policy.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )

        prefix_outputs = outputs_embeds[0].to(torch.float32)
    else:
        prefix_outputs = prefix_embs
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
    train_metrics.policy_update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def update_policy_with_dual_distill(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    teacher_batch: Dict[str, Any] | None,
    optimized_batch: Dict[str, Any],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    *,
    lambda_w2: float,
    lambda_opt: float,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    optimized_action = optimized_batch.get("action")
    if not isinstance(optimized_action, torch.Tensor):
        raise ValueError("optimized_batch must contain tensor `action` for phase-2 distillation.")
    use_teacher_anchor = teacher_batch is not None
    base_action = optimized_action
    if use_teacher_anchor:
        maybe_teacher_action = teacher_batch.get("action")
        if not isinstance(maybe_teacher_action, torch.Tensor):
            raise ValueError("teacher_batch must contain tensor `action` when teacher anchor is enabled.")
        base_action = maybe_teacher_action
    model = getattr(policy, "model", None)
    noise_shape = tuple(base_action.shape)
    max_action_dim = getattr(getattr(policy, "config", None), "max_action_dim", None)
    if isinstance(max_action_dim, int) and base_action.ndim >= 3 and max_action_dim > 0:
        # policy.forward() pads action dim to max_action_dim before using noise/time.
        noise_shape = (*base_action.shape[:-1], int(max_action_dim))

    if model is not None and hasattr(model, "sample_noise") and hasattr(model, "sample_time"):
        noise = model.sample_noise(noise_shape, base_action.device)
        time_tensor = model.sample_time(base_action.shape[0], base_action.device)
    else:
        noise = torch.randn(noise_shape, device=base_action.device, dtype=base_action.dtype)
        time_tensor = torch.rand(base_action.shape[0], device=base_action.device, dtype=base_action.dtype)

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        if use_teacher_anchor:
            loss_w2, _ = policy.forward(teacher_batch, noise=noise, time=time_tensor)
        else:
            loss_w2 = torch.zeros((), device=base_action.device, dtype=base_action.dtype)
        loss_opt, _ = policy.forward(optimized_batch, noise=noise, time=time_tensor)
        loss = float(lambda_opt) * loss_opt
        if use_teacher_anchor:
            loss = loss + float(lambda_w2) * loss_w2

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    total_loss = loss.detach().item()
    w2_loss = loss_w2.detach().item()
    opt_loss = loss_opt.detach().item()
    train_metrics.loss = total_loss
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.policy_update_s = time.perf_counter() - start_time
    output_dict = {
        "distill/loss_w2": w2_loss,
        "distill/loss_opt": opt_loss,
        "distill/loss_total": total_loss,
        "distill/phase2_teacher_anchor_enabled": float(use_teacher_anchor),
    }
    return train_metrics, output_dict


def _infer_policy_action_dim(policy: PreTrainedPolicy) -> int | None:
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return None

    action_feature = getattr(cfg, "action_feature", None)
    shape = getattr(action_feature, "shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) > 0:
        return int(shape[0])

    output_features = getattr(cfg, "output_features", None)
    if isinstance(output_features, dict):
        action_spec = output_features.get("action")
        if action_spec is not None:
            shape = getattr(action_spec, "shape", None)
            if shape is None and isinstance(action_spec, dict):
                shape = action_spec.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                return int(shape[0])
    return None


def _enable_student_padded_action_loss_mask(
    policy: PreTrainedPolicy,
    *,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    if getattr(policy, "_mask_padded_action_loss_enabled", False):
        return getattr(policy, "_mask_padded_action_dim", None)

    action_dim = _infer_policy_action_dim(policy)
    max_action_dim = getattr(getattr(policy, "config", None), "max_action_dim", None)
    if not isinstance(action_dim, int) or action_dim <= 0:
        logging.warning("mask_padded_action_loss enabled but action dim could not be inferred; skipping.")
        return None
    if isinstance(max_action_dim, int) and action_dim >= max_action_dim:
        logging.info(
            "mask_padded_action_loss enabled but action_dim (%s) >= max_action_dim (%s); no-op.",
            action_dim,
            max_action_dim,
        )
        return action_dim

    original_forward = policy.forward

    def _forward_with_action_dim_mask(self, batch: dict[str, torch.Tensor], noise=None, time=None):
        loss, output_dict = original_forward(batch, noise=noise, time=time)
        if not isinstance(output_dict, dict):
            return loss, output_dict

        losses = output_dict.get("losses_after_in_ep_bound")
        if not isinstance(losses, torch.Tensor):
            losses = output_dict.get("losses_after_forward")
        if not isinstance(losses, torch.Tensor):
            return loss, output_dict
        if losses.ndim < 3 or losses.shape[-1] <= action_dim:
            return loss, output_dict

        masked_losses = losses[..., :action_dim]
        masked_loss = masked_losses.mean()
        updated = dict(output_dict)
        updated["loss"] = float(masked_loss.detach().item())
        return masked_loss, updated

    policy.forward = MethodType(_forward_with_action_dim_mask, policy)
    setattr(policy, "_mask_padded_action_loss_enabled", True)
    setattr(policy, "_mask_padded_action_dim", action_dim)
    return action_dim


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
    if not cfg.critic.enable:
        raise ValueError("VGAS and VGAS+ always train the critic; critic.enable must be True.")
    distill_cfg = getattr(cfg, "distill", None)
    distill_enabled = bool(distill_cfg is not None and getattr(distill_cfg, "enable", False))
    phase1_steps = int(max(0, getattr(distill_cfg, "phase1_steps", 5000))) if distill_enabled else 0
    phase1_train_student = bool(getattr(distill_cfg, "phase1_train_student", False)) if distill_enabled else False
    phase1_use_student_as_source = bool(getattr(distill_cfg, "phase1_use_student_as_source", False)) if distill_enabled else False
    phase2_global_samples = int(max(0, getattr(distill_cfg, "phase2_global_samples", 8))) if distill_enabled else 1
    local_samples = int(max(1, getattr(distill_cfg, "local_samples", 4))) if distill_enabled else 1
    phase2_include_dataset_action = bool(
        getattr(distill_cfg, "phase2_include_dataset_action_in_global_pool", False)
    ) if distill_enabled else False
    phase2_local_opt_steps = int(max(0, getattr(distill_cfg, "phase2_local_opt_steps", 5))) if distill_enabled else 0
    phase2_local_opt_lr = float(getattr(distill_cfg, "phase2_local_opt_lr", 3e-4)) if distill_enabled else 0.0
    phase2_local_grad_normalize = bool(getattr(distill_cfg, "phase2_local_grad_normalize", True)) if distill_enabled else True
    phase2_local_adv_weight = bool(getattr(distill_cfg, "phase2_local_adv_weight", False)) if distill_enabled else False
    phase2_local_adv_eps = float(getattr(distill_cfg, "phase2_local_adv_eps", 1e-6)) if distill_enabled else 1e-6
    critic_updates_per_step = max(1, int(getattr(cfg.critic, "critic_updates_per_step", 1)))
    critic_scheduler_step_mode = str(getattr(cfg.critic, "scheduler_step_mode", "critic_update"))
    if critic_scheduler_step_mode not in {"critic_update", "outer_step"}:
        raise ValueError(
            "critic.scheduler_step_mode must be one of {'critic_update', 'outer_step'}, "
            f"got {critic_scheduler_step_mode!r}."
        )
    if distill_enabled:
        phase1_critic_updates_raw = getattr(distill_cfg, "phase1_critic_updates_per_step", None)
        if phase1_critic_updates_raw is None:
            phase1_critic_updates_per_step = critic_updates_per_step
        else:
            phase1_critic_updates_per_step = max(1, int(phase1_critic_updates_raw))
    else:
        phase1_critic_updates_per_step = critic_updates_per_step
    phase1_teacher_action_samples = 1
    if distill_enabled:
        phase1_teacher_action_samples = getattr(distill_cfg, "phase1_teacher_action_samples", None)
        if phase1_teacher_action_samples is None:
            phase1_teacher_action_samples = getattr(distill_cfg, "action_samples", None)
        if phase1_teacher_action_samples is None:
            phase1_teacher_action_samples = 1
        phase1_teacher_action_samples = int(max(1, phase1_teacher_action_samples))
    phase2_disable_critic_best_of_n = bool(
        getattr(distill_cfg, "phase2_disable_critic_best_of_n", False)
    ) if distill_enabled else False
    phase2_use_teacher_anchor = bool(
        getattr(distill_cfg, "phase2_use_teacher_anchor", True)
    ) if distill_enabled else True
    base_loss_rank_weight = float(getattr(cfg.critic, "loss_rank_weight", 1.0))
    phase2_loss_rank_weight_raw = (
        getattr(distill_cfg, "phase2_loss_rank_weight", None) if distill_enabled else None
    )
    phase2_loss_rank_weight = (
        None if phase2_loss_rank_weight_raw is None else float(phase2_loss_rank_weight_raw)
    )
    mask_padded_action_loss = bool(getattr(distill_cfg, "mask_padded_action_loss", True)) if distill_enabled else False
    lambda_w2 = float(getattr(distill_cfg, "lambda_w2", 0.2)) if distill_enabled else 1.0
    lambda_opt = float(getattr(distill_cfg, "lambda_opt", 1.0)) if distill_enabled else 1.0
    if distill_enabled and getattr(distill_cfg, "student_num_steps", None) is not None:
        cfg.policy.num_steps = int(distill_cfg.student_num_steps)
    logging.info(pformat(cfg.to_dict()))

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
    
    dataset = _build_reward_augmented_dataset(cfg)
    logging.info("Using RewardAugmentedLeRobotDataset for training.")
   

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.

    logging.info("Creating policy")
    policy = make_local_smolvla_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    masked_action_dim = _enable_student_padded_action_loss_mask(
        policy,
        enabled=distill_enabled and mask_padded_action_loss,
    )
    if not distill_enabled:
        for p in policy.parameters():
            p.requires_grad_(False)
        policy.eval()

    # Teacher is required when:
    # 1) phase-1 critic source is teacher (i.e. not using student as source), or
    # 2) phase-1 actor distillation is enabled, or
    # 3) phase-2 teacher anchor is enabled.
    teacher_required = bool(
        distill_enabled
        and (
            (
                phase1_steps > 0
                and (
                    phase1_train_student
                    or (not phase1_use_student_as_source)
                )
            )
            or phase2_use_teacher_anchor
        )
    )
    teacher_policy = _build_teacher_policy(cfg, dataset.meta) if teacher_required else None
    if teacher_policy is not None:
        logging.info(
            "Distillation enabled: teacher=%s teacher_num_steps=%s student_num_steps=%s",
            getattr(getattr(teacher_policy, "config", None), "pretrained_path", None),
            getattr(getattr(teacher_policy, "config", None), "num_steps", None),
            getattr(cfg.policy, "num_steps", None),
        )
        logging.info(
            "Distill phases: phase1_steps=%s phase2_global_samples=%s local_samples=%s local_steps=%s local_lr=%s "
            "phase1_train_student=%s local_grad_norm=%s local_adv_weight=%s local_adv_eps=%s "
            "phase1_teacher_action_samples=%s "
            "critic_updates_per_step=%s critic_scheduler_step_mode=%s "
            "mask_padded_action_loss=%s action_dim=%s "
            "lambda_w2=%s lambda_opt=%s "
            "phase2_disable_critic_best_of_n=%s phase2_use_teacher_anchor=%s "
            "loss_rank_weight=%s phase2_loss_rank_weight=%s",
            phase1_steps,
            phase2_global_samples,
            local_samples,
            phase2_local_opt_steps,
            phase2_local_opt_lr,
            phase1_train_student,
            phase2_local_grad_normalize,
            phase2_local_adv_weight,
            phase2_local_adv_eps,
            phase1_teacher_action_samples,
            critic_updates_per_step,
            critic_scheduler_step_mode,
            mask_padded_action_loss,
            masked_action_dim,
            lambda_w2,
            lambda_opt,
            phase2_disable_critic_best_of_n,
            phase2_use_teacher_anchor,
            base_loss_rank_weight,
            phase2_loss_rank_weight,
        )
    elif distill_enabled:
        logging.info(
            "Distillation enabled without teacher instantiation "
            "(phase1_steps=%s phase2_use_teacher_anchor=%s loss_rank_weight=%s phase2_loss_rank_weight=%s).",
            phase1_steps,
            phase2_use_teacher_anchor,
            base_loss_rank_weight,
            phase2_loss_rank_weight,
        )

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

    if distill_enabled:
        logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    else:
        optimizer = None
        lr_scheduler = None
        grad_scaler = None

    step = 0  # number of outer critic-training steps

    critic_state_resume_dir: Optional[Path] = None
    if cfg.resume:
        if distill_enabled:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
            if cfg.checkpoint_path is not None:
                critic_state_resume_dir = Path(cfg.checkpoint_path)
        else:
            critic_state_resume_dir = Path(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None
        # VGAS checkpoints save their step beside the critic state.
        if not distill_enabled and cfg.checkpoint_path is not None:
            resume_step = load_critic_step(Path(cfg.checkpoint_path))
            if resume_step is None:
                try:
                    resume_step = int(Path(cfg.checkpoint_path).name)
                except ValueError:
                    resume_step = None
            if resume_step is not None:
                step = resume_step
    elif cfg.critic_init_checkpoint is not None:
        critic_state_resume_dir = Path(cfg.critic_init_checkpoint)
        logging.info("Initializing critic weights from %s", critic_state_resume_dir)

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

    if distill_enabled:
        policy.train()

    train_metrics = {
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    if distill_enabled:
        train_metrics["loss"] = AverageMeter("loss", ":.3f")
        train_metrics["grad_norm"] = AverageMeter("grdn", ":.3f")
        train_metrics["lr"] = AverageMeter("lr", ":0.1e")
        train_metrics["policy_update_s"] = AverageMeter("pol_updt_s", ":.3f")
        train_metrics["distill_best_q"] = AverageMeter("dst_q", ":.3f")
        train_metrics["distill_samples"] = AverageMeter("dst_n", ":.1f")
        train_metrics["distill_w2_loss"] = AverageMeter("dst_w2", ":.4f")
        train_metrics["distill_opt_loss"] = AverageMeter("dst_opt", ":.4f")
        train_metrics["distill_stage"] = AverageMeter("dst_ph", ":.1f")
        train_metrics["distill_global_gt_hit_rate"] = AverageMeter("dst_gt_hit", ":.3f")
        train_metrics["distill_final_gt_selected_rate"] = AverageMeter("dst_gt_sel", ":.3f")
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
            ("ood_loss_total", "ood_loss_t"),
            ("ood_loss_anchor", "ood_loss_a"),
            ("ood_loss_rank", "ood_loss_r"),
            ("ood_q_mean", "ood_q"),
            ("ood_dist_mean", "ood_dist"),
            ("ood_target_mean", "ood_tgt"),
            ("q_val/ood_avg", "q_ood_avg"),
            ("q_val/ood_policy", "q_ood_pol"),
            ("q_val/ood_prec", "q_ood_prec"),
            ("q_val/ood_mix", "q_ood_mix"),
            ("q_val/ood_trunc", "q_ood_trunc"),
            ("dist/policy", "d_pol"),
            ("dist/prec", "d_prec"),
            ("dist/mix", "d_mix"),
            ("dist/trunc", "d_trunc"),
            ("dist/policy_w1111", "d_pol_1111"),
            ("dist/mix_w1111", "d_mix_1111"),
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
    distill_phase_logged: Optional[int] = None
    ######### offline step
    for _ in range(step, cfg.steps):
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

        
        if critic is None:
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
                    use_vlm_backbone_encode=getattr(cfg, "use_vlm_backbone_encode", True),
                ),
                actor=policy,
                freeze_actor=not distill_enabled,
            )
            vgas_policy = VGASPolicy(actor=policy, critic=critic)
            if critic_state_resume_dir is not None:
                print("reusing Critic")
                load_critic_state(critic, critic_state_resume_dir)
                critic_state_resume_dir = None

        critic_metrics = None
        distill_step_metrics: Dict[str, float] = {}
        distill_phase = 0
        if distill_enabled:
            distill_phase = 1 if step < phase1_steps else 2
            if distill_phase_logged != distill_phase:
                logging.info("Switched to distill phase %s at step=%s", distill_phase, step)
                distill_phase_logged = distill_phase
        if vgas_policy is not None:
            critic_start = time.perf_counter()
            warmup_steps = getattr(cfg.critic, "ood_warmup_steps", 0)
            ood_policy_for_critic: Optional[PreTrainedPolicy] = None
            target_policy_for_critic: Optional[PreTrainedPolicy] = policy
            target_action_samples_override: Optional[int] = None
            if distill_phase == 1:
                if phase1_use_student_as_source:
                    ood_policy_for_critic = policy
                    target_policy_for_critic = policy
                else:
                    if teacher_policy is None:
                        raise ValueError(
                            "Phase-1 distillation requires teacher_policy, but no teacher was instantiated. "
                            "Set distill.phase1_steps=0 or provide a teacher."
                        )
                    ood_policy_for_critic = teacher_policy
                    target_policy_for_critic = teacher_policy
            elif distill_phase == 2:
                ood_policy_for_critic = policy
                target_policy_for_critic = policy
                if phase2_disable_critic_best_of_n:
                    target_action_samples_override = 1
            active_loss_rank_weight = base_loss_rank_weight
            if distill_phase == 2 and phase2_loss_rank_weight is not None:
                active_loss_rank_weight = phase2_loss_rank_weight
            cfg.critic.loss_rank_weight = active_loss_rank_weight
            if hasattr(vgas_policy.critic, "cfg"):
                vgas_policy.critic.cfg.loss_rank_weight = active_loss_rank_weight
            current_critic_updates = phase1_critic_updates_per_step if distill_phase == 1 else critic_updates_per_step
            cached_next_encoding = None
            # Do not reuse precomputed next encodings when target-side state/vision encoders
            # can drift across inner critic updates due to soft target updates.
            allow_cached_next_encoding = (
                current_critic_updates > 1
                and not (
                    vgas_policy.critic.use_independent_critic_encoder
                    or vgas_policy.critic.use_critic_state_encoder
                )
            )
            if allow_cached_next_encoding:
                cached_next_encoding = vgas_policy.critic.precompute_next_target_encoding(
                    batch,
                    target_policy=target_policy_for_critic,
                )
            critic_metrics_acc: Dict[str, float] = {}
            for critic_update_idx in range(current_critic_updates):
                step_critic_scheduler = (
                    critic_scheduler_step_mode == "critic_update"
                    or critic_update_idx == current_critic_updates - 1
                )
                critic_metrics_i = vgas_policy.update_critic(
                    batch,
                    current_step=step,
                    ood_warmup_steps=warmup_steps,
                    ood_policy=ood_policy_for_critic,
                    target_action_samples=target_action_samples_override,
                    target_policy=target_policy_for_critic,
                    cached_next_encoding=cached_next_encoding,
                    step_scheduler=step_critic_scheduler,
                )
                for metric_name, metric_value in critic_metrics_i.items():
                    critic_metrics_acc[metric_name] = critic_metrics_acc.get(metric_name, 0.0) + float(metric_value)
            critic_metrics = {
                metric_name: metric_value / float(current_critic_updates)
                for metric_name, metric_value in critic_metrics_acc.items()
            }
            critic_metrics["critic_lr"] = float(vgas_policy.critic.optimizer.param_groups[0]["lr"])
            critic_metrics["critic_update_s"] = time.perf_counter() - critic_start
            critic_metrics["critic_updates_per_step"] = float(current_critic_updates)
            critic_metrics["critic_updates_per_step_default"] = float(critic_updates_per_step)
            critic_metrics["critic_scheduler_outer_step"] = float(critic_scheduler_step_mode == "outer_step")
            critic_metrics["critic_scheduler_steps_per_outer"] = (
                1.0 if critic_scheduler_step_mode == "outer_step" else float(current_critic_updates)
            )
            if distill_phase > 0:
                critic_metrics["distill/stage"] = float(distill_phase)
                critic_metrics["distill/loss_rank_weight"] = float(active_loss_rank_weight)
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

        output_dict: Dict[str, Any] = {}
        if distill_enabled:
            # VGAS+ updates the policy only through its two distillation phases.
            if distill_phase == 1:
                if not phase1_train_student:
                    # Optional ablation: phase-1 is critic-only warmup, skip all student updates.
                    output_dict = {}
                    distill_step_metrics["distill/phase1_student_update_enabled"] = 0.0
                    distill_step_metrics["distill/action_samples"] = 0.0
                else:
                    if teacher_policy is None:
                        raise ValueError(
                            "Phase-1 student distillation requires teacher_policy, but no teacher was instantiated. "
                            "Set phase1_train_student=false for critic-only warmup, or enable teacher loading."
                        )
                    # Phase-1: distill student from teacher actions (optionally re-ranked by critic).
                    distill_samples = int(max(1, phase1_teacher_action_samples))
                    if critic is not None:
                        selected_actions, distill_step_metrics = _select_best_teacher_actions(
                            teacher=teacher_policy,
                            critic=critic,
                            batch=batch,
                            action_samples=distill_samples,
                        )
                    else:
                        # Fallback when critic is unavailable: use teacher single-sample chunk directly.
                        with torch.no_grad():
                            selected_actions = teacher_policy.predict_action_chunk(batch)
                        distill_step_metrics = {
                            "distill/action_samples": float(distill_samples),
                        }
                    distill_step_metrics["distill/phase1_student_update_enabled"] = 1.0
                    actor_batch = _build_distill_actor_batch(batch=batch, selected_actions=selected_actions)
                    train_tracker, output_dict = update_policy(
                        train_tracker,
                        policy,
                        actor_batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        grad_scaler=grad_scaler,
                        lr_scheduler=lr_scheduler,
                        use_amp=cfg.policy.use_amp,
                    )
            elif distill_phase == 2:
                # Phase-2: optional teacher-anchor target + student global/local optimized target.
                if phase2_use_teacher_anchor:
                    if teacher_policy is None:
                        raise ValueError(
                            "phase2_use_teacher_anchor=True but no teacher_policy was instantiated."
                        )
                    teacher_actions = _sample_policy_actions_single(
                        policy=teacher_policy,
                        batch=batch,
                        q_chunk_len=critic.q_chunk_len if critic is not None else None,
                    )
                    teacher_batch = _build_distill_actor_batch(batch=batch, selected_actions=teacher_actions)
                    distill_step_metrics["distill/phase2_teacher_anchor_enabled"] = 1.0
                else:
                    teacher_batch = None
                    distill_step_metrics["distill/phase2_teacher_anchor_enabled"] = 0.0

                if critic is not None:
                    # Build optimized target from student samples using critic guidance.
                    optimized_actions, phase2_metrics = _select_local_optimized_student_actions(
                        student=policy,
                        critic=critic,
                        batch=batch,
                        action_samples=phase2_global_samples,
                        local_samples=local_samples,
                        include_dataset_action_in_global_pool=phase2_include_dataset_action,
                        local_steps=phase2_local_opt_steps,
                        local_lr=phase2_local_opt_lr,
                        local_grad_normalize=phase2_local_grad_normalize,
                        local_adv_weight=phase2_local_adv_weight,
                        local_adv_eps=phase2_local_adv_eps,
                    )
                else:
                    with torch.no_grad():
                        optimized_actions = policy.predict_action_chunk(batch)
                    phase2_metrics = {
                        "distill/global_samples": float(phase2_global_samples),
                        "distill/local_samples": float(1),
                    }
                optimized_batch = _build_distill_actor_batch(batch=batch, selected_actions=optimized_actions)
                train_tracker, output_dict = update_policy_with_dual_distill(
                    train_tracker,
                    policy,
                    teacher_batch,
                    optimized_batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    grad_scaler=grad_scaler,
                    lambda_w2=lambda_w2,
                    lambda_opt=lambda_opt,
                    lr_scheduler=lr_scheduler,
                    use_amp=cfg.policy.use_amp,
                )
                distill_step_metrics.update(phase2_metrics)
                distill_step_metrics.update(output_dict)
        if distill_enabled:
            if "distill_stage" in train_tracker.metrics:
                setattr(train_tracker, "distill_stage", float(distill_phase))
            if distill_phase == 1:
                if "distill_best_q" in train_tracker.metrics and "distill/best_q_mean" in distill_step_metrics:
                    setattr(train_tracker, "distill_best_q", distill_step_metrics["distill/best_q_mean"])
                if "distill_samples" in train_tracker.metrics and "distill/action_samples" in distill_step_metrics:
                    setattr(train_tracker, "distill_samples", distill_step_metrics["distill/action_samples"])
                if "distill_w2_loss" in train_tracker.metrics:
                    setattr(train_tracker, "distill_w2_loss", 0.0)
                if "distill_opt_loss" in train_tracker.metrics:
                    setattr(train_tracker, "distill_opt_loss", 0.0)
            elif distill_phase == 2:
                if "distill_best_q" in train_tracker.metrics and "distill/local_best_q_mean" in distill_step_metrics:
                    setattr(train_tracker, "distill_best_q", distill_step_metrics["distill/local_best_q_mean"])
                if "distill_samples" in train_tracker.metrics and "distill/global_samples" in distill_step_metrics:
                    setattr(train_tracker, "distill_samples", distill_step_metrics["distill/global_samples"])
                if (
                    "distill_global_gt_hit_rate" in train_tracker.metrics
                    and "distill/global_gt_hit_rate" in distill_step_metrics
                ):
                    setattr(train_tracker, "distill_global_gt_hit_rate", distill_step_metrics["distill/global_gt_hit_rate"])
                if (
                    "distill_final_gt_selected_rate" in train_tracker.metrics
                    and "distill/final_gt_selected_rate" in distill_step_metrics
                ):
                    setattr(
                        train_tracker,
                        "distill_final_gt_selected_rate",
                        distill_step_metrics["distill/final_gt_selected_rate"],
                    )
                if "distill_w2_loss" in train_tracker.metrics and "distill/loss_w2" in distill_step_metrics:
                    setattr(train_tracker, "distill_w2_loss", distill_step_metrics["distill/loss_w2"])
                if "distill_opt_loss" in train_tracker.metrics and "distill/loss_opt" in distill_step_metrics:
                    setattr(train_tracker, "distill_opt_loss", distill_step_metrics["distill/loss_opt"])

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if "dist/policy" in wandb_log_dict:
                    wandb_log_dict["dist/policy_avg"] = wandb_log_dict["dist/policy"]
                if "dist/mix" in wandb_log_dict:
                    wandb_log_dict["dist/mix_avg"] = wandb_log_dict["dist/mix"]
                if "dist/policy_w1111" in wandb_log_dict:
                    wandb_log_dict["dist/policy_w1111_avg"] = wandb_log_dict["dist/policy_w1111"]
                if "dist/mix_w1111" in wandb_log_dict:
                    wandb_log_dict["dist/mix_w1111_avg"] = wandb_log_dict["dist/mix_w1111"]
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if critic_metrics:
                    wandb_log_dict.update(critic_metrics)
                if distill_step_metrics:
                    wandb_log_dict.update(distill_step_metrics)
                wandb_logger.log_dict(_sanitize_for_wandb(wandb_log_dict), step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            checkpoint_label = "VGAS+ policy and critic" if distill_enabled else "VGAS critic"
            logging.info("Checkpoint %s after step %s", checkpoint_label, step)
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            if distill_enabled:
                save_checkpoint(
                    checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor
                )
                if wandb_logger and cfg.log_policy_to_wandb:
                    wandb_logger.log_policy(checkpoint_dir)
            save_critic_state(critic, checkpoint_dir)
            save_critic_step(step, checkpoint_dir)
            update_last_checkpoint(checkpoint_dir)

def _critic_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / TRAINING_STATE_DIR / CRITIC_STATE_FILE


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


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
        cfg_dict = asdict(cfg)
        payload["critic_config"] = cfg_dict
        config_path = critic_path / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(cfg_dict), f, indent=2, sort_keys=True)
    torch.save(payload, last_state)


def _resolve_critic_state_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = [
        path / "critic_pretrained_model" / "last.ckpt",
        path / "last.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path / "critic_pretrained_model" / "last.ckpt"


def load_critic_state(critic: Optional[QChunkedCritic], checkpoint_dir: Path) -> None:
    if critic is None:
        return
    critic_path = _resolve_critic_state_path(checkpoint_dir)
    if critic_path.exists():
        payload = torch.load(critic_path, map_location=critic.device, weights_only=False)
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
        cur_idx = torch.cuda.current_device()        # index within this process (0..N-1)
        print("current_device index =", cur_idx)
        print("current_device name  =", torch.cuda.get_device_name(cur_idx))

if __name__ == "__main__":
    main()
