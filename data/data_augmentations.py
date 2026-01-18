"""
V-GPS style visual augmentations (resize -> random_resized_crop -> color jitter).

This mirrors the order/params from V-GPS `experiments/configs/data_config.py`:
resize to 256x256, then random_resized_crop with scale [0.8, 1.0] and ratio [0.9, 1.1],
followed by brightness/contrast/saturation/hue jitter.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _random_resized_crop(
    img: torch.Tensor,
    size: Sequence[int] | torch.Size,
    scale: Iterable[float] = (0.8, 1.0),
    ratio: Iterable[float] = (0.9, 1.1),
) -> torch.Tensor:
    """Torch implementation of RandomResizedCrop for a single CHW image tensor in [0, 1]."""

    _, height, width = img.shape
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio, device=img.device, dtype=torch.float32))

    for _ in range(10):
        target_area = area * float(torch.empty(1, device=img.device).uniform_(scale[0], scale[1]))
        aspect = float(torch.exp(torch.empty(1, device=img.device).uniform_(log_ratio[0], log_ratio[1])))
        w_crop = int(round(math.sqrt(target_area * aspect)))
        h_crop = int(round(math.sqrt(target_area / aspect)))

        if 0 < w_crop <= width and 0 < h_crop <= height:
            top = int(torch.randint(0, height - h_crop + 1, (1,), device=img.device).item())
            left = int(torch.randint(0, width - w_crop + 1, (1,), device=img.device).item())
            cropped = img[:, top : top + h_crop, left : left + w_crop]
            return TF.resize(cropped, size, antialias=True)

    # Fallback to a centered crop if we failed to sample a valid one.
    in_ratio = width / height
    if in_ratio < min(ratio):
        w_crop = width
        h_crop = int(round(w_crop / min(ratio)))
    elif in_ratio > max(ratio):
        h_crop = height
        w_crop = int(round(h_crop * max(ratio)))
    else:
        w_crop, h_crop = width, height

    top = max(0, (height - h_crop) // 2)
    left = max(0, (width - w_crop) // 2)
    cropped = img[:, top : top + h_crop, left : left + w_crop]
    return TF.resize(cropped, size, antialias=True)


def _color_jitter(
    img: torch.Tensor,
    brightness: float = 0.1,
    contrast: tuple[float, float] = (0.9, 1.1),
    saturation: tuple[float, float] = (0.9, 1.1),
    hue: float = 0.05,
) -> torch.Tensor:
    """Color jitter matching V-GPS config (additive brightness, contrast/saturation multiplicative, hue shift)."""

    # Brightness: additive delta in [-brightness, brightness].
    if brightness > 0:
        delta = float(torch.empty(1, device=img.device).uniform_(-brightness, brightness))
        img = torch.clamp(img + delta, 0.0, 1.0)

    # Contrast: scale relative to mean.
    if contrast is not None and len(contrast) == 2:
        factor = float(torch.empty(1, device=img.device).uniform_(contrast[0], contrast[1]))
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = torch.clamp((img - mean) * factor + mean, 0.0, 1.0)

    # Saturation.
    if saturation is not None and len(saturation) == 2:
        sat_factor = float(torch.empty(1, device=img.device).uniform_(saturation[0], saturation[1]))
        img = torch.clamp(TF.adjust_saturation(img, sat_factor), 0.0, 1.0)

    # Hue shift in [-hue, hue].
    if hue > 0:
        hue_factor = float(torch.empty(1, device=img.device).uniform_(-hue, hue))
        img = torch.clamp(TF.adjust_hue(img, hue_factor), 0.0, 1.0)

    return img


def vgps_augment(
    images: torch.Tensor,
    image_size: Sequence[int] | torch.Size = (256, 256),
    random_resized_crop_scale: Iterable[float] = (0.8, 1.0),
    random_resized_crop_ratio: Iterable[float] = (0.9, 1.1),
    brightness: float = 0.2,
    contrast: tuple[float, float] = (0.8, 1.2),
    saturation: tuple[float, float] = (0.8, 1.2),
    hue: float = 0.05,
) -> torch.Tensor:
    """
    Apply V-GPS augmentations to a batch of images in [0, 1].

    Args:
        images: Tensor shaped (B, C, H, W) or (C, H, W), values in [0, 1].
        image_size: Final resize target (height, width).
        random_resized_crop_scale: Scale range for random resized crop.
        random_resized_crop_ratio: Aspect ratio range for random resized crop.
        brightness/contrast/saturation/hue: Color jitter params.
    """

    single = False
    if images.ndim == 3:
        images = images.unsqueeze(0)
        single = True
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped (B,C,H,W) or (C,H,W), got {images.shape}")

    augmented = []
    for img in images:
        # Resize first to match V-GPS preprocessing.
        img_resized = TF.resize(img, image_size, antialias=True)
        cropped = _random_resized_crop(
            img_resized, image_size, scale=random_resized_crop_scale, ratio=random_resized_crop_ratio
        )
        jittered = _color_jitter(cropped, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        augmented.append(jittered)

    out = torch.stack(augmented, dim=0)
    if single:
        out = out.squeeze(0)
    return out


def vgps_augment_vmap(
    images: torch.Tensor,
    image_size: Sequence[int] | torch.Size = (256, 256),
    random_resized_crop_scale: Iterable[float] = (0.9, 1.0),
    random_resized_crop_ratio: Iterable[float] = (0.9, 1.1),
    brightness: float = 0.25,
    contrast: tuple[float, float] = (0.9, 1.1),
    saturation: tuple[float, float] = (0.9, 1.1),
    hue: float = 0.05,
) -> torch.Tensor:
    """
    Vectorized variant using torch.vmap when available; falls back to loop if not.
    Behavior matches vgps_augment.
    """

    single = False
    if images.ndim == 3:
        images = images.unsqueeze(0)
        single = True
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped (B,C,H,W) or (C,H,W), got {images.shape}")

    def _augment_one(img: torch.Tensor) -> torch.Tensor:
        img_resized = TF.resize(img, image_size, antialias=True)
        cropped = _random_resized_crop(
            img_resized, image_size, scale=random_resized_crop_scale, ratio=random_resized_crop_ratio
        )
        return _color_jitter(cropped, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    out = None
    if hasattr(torch, "vmap"):
        try:
            # Torch vmap requires randomness="different" to allow random ops.
            out = torch.vmap(_augment_one, randomness="different")(images)  # type: ignore[arg-type]
        except TypeError:
            # Older torch without randomness kwarg.
            out = torch.vmap(_augment_one)(images)
        except RuntimeError:
            out = None
    if out is None:
        out = vgps_augment(
            images,
            image_size=image_size,
            random_resized_crop_scale=random_resized_crop_scale,
            random_resized_crop_ratio=random_resized_crop_ratio,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    if single:
        out = out.squeeze(0)
    return out


__all__ = ["vgps_augment", "vgps_augment_vmap"]


if __name__ == "__main__":
    # Quick smoke test: run augment on random images and print shapes/ranges.
    torch.manual_seed(0)
    dummy = torch.rand(2, 3, 256, 256)
    out = vgps_augment(dummy)
    out_vmap = vgps_augment_vmap(dummy)
    print("Loop output shape:", out.shape, "min/max:", float(out.min()), float(out.max()))
    print("vmap output shape:", out_vmap.shape, "min/max:", float(out_vmap.min()), float(out_vmap.max()))
