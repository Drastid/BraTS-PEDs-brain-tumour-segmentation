"""
src/train_utils.py
==================
Training and evaluation utilities for the BraTS-PEDs segmentation pipeline.

Public API
----------
get_augmentation()      → albumentations.Compose  (geometric-only)
get_class_weights()     → torch.Tensor            (inverse-frequency weights)
train_one_epoch()       → dict[str, float]        (loss + per-class Dice)
evaluate()              → dict[str, float]        (loss + per-class Dice)
set_encoder_trainable() → None                    (freeze / unfreeze encoder)
save_checkpoint()       → None
load_checkpoint()       → dict                    (state restored)
MetricTracker           → class                   (running-mean bookkeeper)

Augmentation policy
-------------------
Only GEOMETRIC transforms are used.  Pixel-value augmentations
(brightness, contrast, colour jitter) are explicitly excluded because
the input tensors are Z-score-normalised (contain negative values) and
any pixel-level modification would corrupt the normalisation.

Allowed transforms (albumentations):
    HorizontalFlip, RandomRotate90, ShiftScaleRotate, ElasticTransform
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CLASS_NAMES, NUM_CLASSES, VOXEL_FREQ
from .losses import CombinedLoss


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def get_augmentation(p: float = 0.5) -> A.Compose:
    """Return a geometric-only albumentations transform pipeline.

    IMPORTANT — only geometric augmentations are included.
    Pixel-level transforms (RandomBrightnessContrast, ColorJitter, etc.)
    are intentionally omitted to preserve Z-score-normalised float32 values.

    Args:
        p: Probability applied to each individual transform (default 0.5).

    Returns:
        albumentations.Compose pipeline compatible with
        ``BraTSDataset`` (accepts ``image`` HWC float32, ``mask`` HW uint8).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.RandomRotate90(p=p),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=0,          # cv2.BORDER_CONSTANT — fills with 0
                p=p,
            ),
            A.ElasticTransform(
                alpha=30.0,
                sigma=5.0,
                p=p * 0.5,              # applied less frequently (heavy op)
            ),
        ],
        additional_targets={"mask": "mask"},
    )


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def get_class_weights(
    voxel_freq: Optional[np.ndarray] = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency class weights for loss functions.

    weight_c = (1 / freq_c) / Σ (1 / freq_c')

    The resulting tensor sums to 1 and up-weights rare tumour classes
    (NCR, ET) heavily relative to the near-ubiquitous background.

    Args:
        voxel_freq: 1-D array of per-class voxel frequencies (must sum ≈ 1).
                    Defaults to the BraTS-PEDs global statistics from EDA.
        device:     Target device for the returned tensor.

    Returns:
        Float tensor of shape [NUM_CLASSES].
    """
    freq = voxel_freq if voxel_freq is not None else VOXEL_FREQ
    inv  = 1.0 / (freq + 1e-8)
    w    = inv / inv.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------


class MetricTracker:
    """Accumulates running means for an arbitrary set of named scalars.

    Usage::

        tracker = MetricTracker()
        for batch in loader:
            tracker.update({"loss": 0.42, "dice_ncr": 0.31})
        means = tracker.means()   # {"loss": …, "dice_ncr": …}
        tracker.reset()

    Thread-safety is NOT guaranteed (single-process training assumed).
    """

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        """Add a batch of metric values.

        Args:
            metrics: Mapping from metric name to scalar value.
            n:       Batch size (used as weight for the running mean).
        """
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0)   + float(v) * n
            self._counts[k] = self._counts.get(k, 0)   + n

    def means(self) -> Dict[str, float]:
        """Return the running mean for each tracked metric."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }


# ---------------------------------------------------------------------------
# Per-class Dice (non-differentiable, for logging)
# ---------------------------------------------------------------------------


def _compute_batch_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-class Dice coefficient on a single batch (no grad).

    Args:
        logits:      [B, C, H, W] float tensor.
        targets:     [B, H, W]    long tensor.
        num_classes: Number of classes.
        smooth:      Laplace smoothing term.

    Returns:
        Dict mapping ``"dice_<classname>"`` → float in [0, 1].
    """
    with torch.no_grad():
        preds   = logits.argmax(dim=1)                              # [B, H, W]
        results = {}
        for c in range(num_classes):
            pred_c = (preds   == c).float()
            true_c = (targets == c).float()
            inter  = (pred_c * true_c).sum()
            denom  = pred_c.sum() + true_c.sum()
            dice   = (2.0 * inter + smooth) / (denom + smooth)
            results[f"dice_{CLASS_NAMES[c]}"] = dice.item()
    return results


# ---------------------------------------------------------------------------
# Center-crop helper
# ---------------------------------------------------------------------------


def center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic center crop applied to image and mask tensors.

    Args:
        image:     [B, C, H, W] float tensor.
        mask:      [B, H, W]    long tensor.
        crop_size: Side length of the square crop (default 192).

    Returns:
        Cropped (image, mask) pair with spatial dims [crop_size, crop_size].
    """
    _, _, H, W = image.shape
    top  = (H - crop_size) // 2
    left = (W - crop_size) // 2
    image_crop = image[:, :, top:top + crop_size, left:left + crop_size]
    mask_crop  = mask[:,    top:top + crop_size, left:left + crop_size]
    return image_crop, mask_crop


# ---------------------------------------------------------------------------
# Encoder freeze / unfreeze
# ---------------------------------------------------------------------------


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the encoder backbone of a SMP model.

    Segmentation-models-pytorch models expose the encoder as
    ``model.encoder``.  This function iterates over its parameters and
    sets ``requires_grad`` accordingly.

    Args:
        model:     SMP segmentation model (Unet, FPN, …).
        trainable: True  → encoder parameters are updated by the optimiser.
                   False → encoder parameters are frozen (no gradient).
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError(
            "Model has no '.encoder' attribute. "
            "Ensure you are using a segmentation-models-pytorch model."
        )
    for param in encoder.parameters():
        param.requires_grad = trainable

    state = "trainable" if trainable else "frozen"
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[set_encoder_trainable] Encoder is now {state} ({n_params:,} params).")


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    crop_size: int = 192,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Run one full training epoch.

    Args:
        model:     Segmentation model in train mode.
        loader:    Training DataLoader.
        criterion: CombinedLoss instance.
        optimizer: Optimiser (e.g. AdamW).
        device:    Computation device.
        crop_size: Side length of the center crop applied before forwarding.
        scaler:    Optional AMP GradScaler for mixed-precision training.
                   Pass ``None`` to disable AMP.

    Returns:
        Dict of mean metrics over the epoch:
        ``loss``, ``dice_loss``, ``focal_loss``, ``dice_<class>`` × 4.
    """
    model.train()
    tracker = MetricTracker()
    use_amp = scaler is not None

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)   # [B, 4, H, W]
        masks  = masks.to(device,  non_blocking=True)   # [B, H, W]

        images, masks = center_crop(images, masks, crop_size)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=str(device).split(":")[0], enabled=use_amp):
            logits = model(images)                      # [B, C, H, W]
            total, d_loss, f_loss = criterion(logits, masks)

        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_metrics = {
            "loss":       total.item(),
            "dice_loss":  d_loss.item(),
            "focal_loss": f_loss.item(),
        }
        batch_metrics.update(_compute_batch_dice(logits, masks))
        tracker.update(batch_metrics, n=images.size(0))

    return tracker.means()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device | str,
    crop_size: int = 192,
) -> Dict[str, float]:
    """Evaluate the model on a validation or test DataLoader.

    Args:
        model:     Segmentation model (will be set to eval mode).
        loader:    Validation / test DataLoader.
        criterion: CombinedLoss instance.
        device:    Computation device.
        crop_size: Side length of the center crop (must match training).

    Returns:
        Dict of mean metrics over the full loader:
        ``loss``, ``dice_loss``, ``focal_loss``, ``dice_<class>`` × 4.
    """
    model.eval()
    tracker = MetricTracker()

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        images, masks = center_crop(images, masks, crop_size)

        logits = model(images)
        total, d_loss, f_loss = criterion(logits, masks)

        batch_metrics = {
            "loss":       total.item(),
            "dice_loss":  d_loss.item(),
            "focal_loss": f_loss.item(),
        }
        batch_metrics.update(_compute_batch_dice(logits, masks))
        tracker.update(batch_metrics, n=images.size(0))

    return tracker.means()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    metrics: Dict[str, float],
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    """Serialise training state to disk.

    Saved keys: ``model``, ``optimizer``, ``scheduler``, ``epoch``,
    ``metrics``, and optionally ``scaler`` (for AMP).

    Args:
        path:      Destination file path (e.g. ``checkpoints/best.pth``).
        model:     Model whose ``state_dict`` will be saved.
        optimizer: Optimiser state.
        scheduler: LR scheduler state.
        epoch:     Current epoch index (0-based).
        metrics:   Dict of metric values to store alongside the weights.
        scaler:    Optional AMP GradScaler.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device | str = "cpu",
) -> Dict:
    """Restore training state from a checkpoint file.

    Args:
        path:      Path to the ``.pth`` checkpoint.
        model:     Model to restore weights into.
        optimizer: Optional optimiser to restore state into.
        scheduler: Optional LR scheduler to restore state into.
        scaler:    Optional AMP GradScaler to restore state into.
        device:    Map location for tensor loading.

    Returns:
        The full checkpoint dict (contains ``epoch``, ``metrics``, etc.).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer  is not None and "optimizer"  in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  is not None and "scheduler"  in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler     is not None and "scaler"     in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format a metrics dict into a compact log string.

    Example output::

        [val]  loss=0.4231  dice_NCR=0.512  dice_ED=0.731  dice_ET=0.623

    Args:
        metrics: Dict returned by ``train_one_epoch`` or ``evaluate``.
        prefix:  Optional label prepended to the string.

    Returns:
        Formatted string.
    """
    parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
    body  = "  ".join(parts)
    return f"[{prefix}]  {body}" if prefix else body
