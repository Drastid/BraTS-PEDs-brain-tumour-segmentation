"""
src/train_utils.py
==================
Training and evaluation utilities for the BraTS-PEDs segmentation pipeline.

Public API
----------
set_seed()              → None                    (seed RNGs + cuDNN determinism)
seed_worker()           → None                    (DataLoader worker_init_fn)
make_generator()        → torch.Generator         (seeded DataLoader generator)
get_augmentation()      → albumentations.Compose  (geometric-only; optional DTM)
get_class_weights()     → torch.Tensor            (inverse-frequency weights)
train_one_epoch()       → dict[str, float]        (loss + per-class Dice)
evaluate()              → dict[str, float]        (loss + per-class Dice)
train_one_epoch_gsl()   → dict[str, float]        (scheduled Dice-Focal + GSL)
evaluate_gsl()          → dict[str, float]        (scheduled Dice-Focal + GSL)
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
import random
import time
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import (
    CLASS_NAMES,
    CROP_SIZE,
    NUM_CLASSES,
    VOXEL_FREQ,
    compute_crop_offsets,
)
from .losses import CombinedLoss, DiceFocalGSLLoss


# ---------------------------------------------------------------------------
# Reproducibility (review.md §6.3)
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) enable deterministic cuDNN.

    Seeds ``random``, ``numpy``, and ``torch`` (CPU + all CUDA devices). When
    ``deterministic=True`` it also forces cuDNN into a reproducible mode so that
    repeated runs — and therefore the model/ablation comparisons in the
    project plan — are actually comparable.

    Args:
        seed:          Master seed (default 42, matching the dataset split).
        deterministic: If True, set ``cudnn.deterministic=True`` and
                       ``cudnn.benchmark=False``. This trades a little speed for
                       reproducibility; set False only when you explicitly want
                       cuDNN autotuning and accept run-to-run variance.

    Notes:
        DataLoader worker processes are seeded separately via
        :func:`seed_worker` together with a seeded :func:`make_generator`
        passed as the loader's ``generator``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that makes per-worker RNG reproducible.

    Each worker derives its NumPy/``random`` seed from ``torch.initial_seed()``,
    which PyTorch sets deterministically per worker from the loader's
    ``generator``. Without this, ``num_workers > 0`` makes augmentation order
    non-reproducible across runs even with a fixed master seed.

    Pass this as ``DataLoader(..., worker_init_fn=seed_worker,
    generator=make_generator(seed))``.
    """
    del worker_id  # signature mandated by PyTorch; the id itself is unused
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling.

    Passing this as ``DataLoader(generator=...)`` fixes the shuffle order and,
    combined with :func:`seed_worker`, makes the whole input pipeline
    deterministic.

    Args:
        seed: Seed for the generator (default 42).

    Returns:
        A CPU ``torch.Generator`` seeded with ``seed``.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def get_augmentation(p: float = 0.5, with_dtm: bool = False) -> A.Compose:
    """Return a geometric-only albumentations transform pipeline.

    IMPORTANT — only geometric augmentations are included.
    Pixel-level transforms (RandomBrightnessContrast, ColorJitter, etc.)
    are intentionally omitted to preserve Z-score-normalised float32 values.

    Two regimes (controlled by ``with_dtm``):

    * ``with_dtm=False`` (baseline models — Dice/Focal only):
      the full geometric set is used, including ``ShiftScaleRotate`` and
      ``ElasticTransform``. These continuous deformations are fine here because
      the only spatial targets are the image and the integer mask.

    * ``with_dtm=True`` (Generalized Surface Loss):
      the pipeline is restricted to **rigid, orthogonal** transforms only —
      ``HorizontalFlip``, ``VerticalFlip``, ``RandomRotate90``. Continuous
      deformations (``ShiftScaleRotate``) and elastic warping
      (``ElasticTransform``) are categorically EXCLUDED.

      Rationale (review of the GSL implementation): the Distance Transform Map
      encodes the *true Euclidean distance* of every voxel from the object
      boundary. Rigid 90°-multiple rotations and axis flips are isometries —
      they permute voxels without changing inter-voxel distances, so the
      transformed DTM is still an exact DTM. Scaling, shear, sub-pixel rotation
      and elastic warping, however, change those distances and require
      interpolation, which corrupts the DTM and therefore pollutes the
      boundary-aware ("surgical") gradient the GSL is designed to provide. To
      keep the DTM mathematically valid we drop them entirely in this regime.

    Args:
        p:        Probability applied to each individual transform (default 0.5).
        with_dtm: If True, register a ``dtm`` additional target of type
                  ``image`` AND restrict the pipeline to DTM-preserving rigid
                  transforms (see above).

    Returns:
        albumentations.Compose pipeline compatible with
        ``BraTSDataset`` (accepts ``image`` HWC float32, ``mask`` HW uint8,
        and optionally ``dtm`` HWC float32).
    """
    additional_targets = {"mask": "mask"}

    if with_dtm:
        # DTM-safe regime: rigid orthogonal transforms only (isometries).
        # No ShiftScaleRotate / ElasticTransform — they would invalidate the DTM.
        additional_targets["dtm"] = "image"
        transforms = [
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.RandomRotate90(p=p),
        ]
    else:
        # Baseline regime: full geometric set (image + integer mask only).
        transforms = [
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
        ]

    return A.Compose(transforms, additional_targets=additional_targets)


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
# AMP / device helpers
# ---------------------------------------------------------------------------


def resolve_device_type(device: torch.device | str) -> str:
    """Return the bare device type ("cuda" | "cpu" | "mps") for autocast.

    Robust against ``torch.device`` objects and indexed strings ("cuda:0").
    Previously this was done inline with ``str(device).split(":")[0]``, which
    is fragile; centralising it keeps ``train_one_epoch`` and ``evaluate``
    consistent.

    Args:
        device: A ``torch.device`` or device string.

    Returns:
        The device *type* string, e.g. ``"cuda"``.
    """
    return torch.device(device).type


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    """Map a config string to a torch autocast dtype.

    Args:
        amp_dtype: One of ``"fp16"``, ``"bf16"``. (``"none"`` is handled by the
                   caller by disabling autocast, not by this function.)

    Returns:
        ``torch.float16`` or ``torch.bfloat16``.

    Raises:
        ValueError: If ``amp_dtype`` is not a recognised value.

    Notes:
        On Ampere+ GPUs (A100) ``"bf16"`` is recommended: it has the same
        exponent range as fp32, so the Dice/Focal reductions are far less
        prone to overflow/underflow than fp16, and it does not require loss
        scaling.
    """
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if amp_dtype not in mapping:
        raise ValueError(
            f"Unsupported amp_dtype {amp_dtype!r}; expected one of "
            f"{sorted(mapping)} (or 'none' to disable AMP)."
        )
    return mapping[amp_dtype]


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
    crop_size: int = CROP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic center crop applied to image and mask tensors.

    Uses :func:`src.constants.compute_crop_offsets` so the offset formula is
    shared with 3D inference (``eval_utils.predict_volume``) — the two must
    agree exactly or predictions shift relative to the ground truth.

    Args:
        image:     [B, C, H, W] float tensor.
        mask:      [B, H, W]    long tensor.
        crop_size: Side length of the square crop (default ``CROP_SIZE``).

    Returns:
        Cropped (image, mask) pair with spatial dims [crop_size, crop_size].
    """
    _, _, H, _ = image.shape
    top, left = compute_crop_offsets(H, crop_size)
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


def _trainable_params(model: nn.Module):
    """Yield only parameters that require gradients.

    Used for gradient clipping so that frozen parameters (e.g. the encoder
    during Phase 1) are excluded from the norm computation (review.md §6.6).
    ``clip_grad_norm_`` already skips ``grad is None``, so this is a precision
    refinement rather than a correctness fix, but it makes the clipped set
    explicit and matches what the optimiser actually updates.
    """
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    crop_size: int = CROP_SIZE,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: str = "fp16",
) -> Dict[str, float]:
    """Run one full training epoch.

    Mixed precision (review.md §6.1, §6.2)
    --------------------------------------
    The model ``forward`` runs under ``autocast`` (fast, memory-light), but the
    **loss is computed in fp32**: ``logits.float()`` is taken *outside* the
    autocast region before calling ``criterion``. The Dice/Focal reductions
    involve softmax, ``log``, and sums over many pixels with a ``1e-6`` epsilon;
    evaluating those in fp16 risks under/overflow. This mirrors the cautious
    pattern already used in the SegFormer notebook's pre-flight check and makes
    all three models numerically consistent.

    With ``amp_dtype="bf16"`` (recommended on A100) loss scaling is unnecessary
    because bfloat16 keeps the fp32 exponent range; pass ``scaler=None`` in that
    case and the function skips the scaler path automatically.

    Args:
        model:     Segmentation model in train mode.
        loader:    Training DataLoader.
        criterion: CombinedLoss instance.
        optimizer: Optimiser (e.g. AdamW).
        device:    Computation device.
        crop_size: Side length of the center crop applied before forwarding.
        scaler:    Optional AMP GradScaler. Required for ``amp_dtype="fp16"``;
                   should be ``None`` for ``"bf16"`` or full precision.
        amp_dtype: Autocast dtype: ``"fp16"``, ``"bf16"``, or ``"none"`` to
                   disable autocast entirely.

    Returns:
        Dict of mean metrics over the epoch:
        ``loss``, ``dice_loss``, ``focal_loss``, ``dice_<class>`` × 4.
    """
    model.train()
    tracker = MetricTracker()

    device_type = resolve_device_type(device)
    use_amp = amp_dtype != "none"
    autocast_dtype = resolve_amp_dtype(amp_dtype) if use_amp else None
    # GradScaler is only meaningful for fp16; bf16/full precision must not scale.
    use_scaler = scaler is not None and scaler.is_enabled() and amp_dtype == "fp16"

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)   # [B, 4, H, W]
        masks  = masks.to(device,  non_blocking=True)   # [B, H, W]

        images, masks = center_crop(images, masks, crop_size)

        optimizer.zero_grad(set_to_none=True)

        # Forward under autocast for speed/memory …
        with torch.amp.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=use_amp
        ):
            logits = model(images)                      # [B, C, H, W]

        # … but compute the loss in fp32 for numerical stability.
        total, d_loss, f_loss = criterion(logits.float(), masks)

        if use_scaler:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
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
    crop_size: int = CROP_SIZE,
) -> Dict[str, float]:
    """Evaluate the model on a validation or test DataLoader.

    Runs at full precision under ``no_grad`` (no autocast), so the loss is
    already numerically stable and needs no fp32 cast.

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
# Generalized Surface Loss training/eval (Celaya et al.)
# ---------------------------------------------------------------------------
#
# Separate from train_one_epoch/evaluate because DiceFocalGSLLoss has a
# different signature (it needs the per-class DTM and an epoch-indexed alpha)
# and returns an extra component. Keeping them separate preserves the original
# CombinedLoss training path unchanged for the baseline models.


def _center_crop_dtm(dtm: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Center-crop a DTM tensor [B, C, H, W] to match images/masks.

    Uses the shared offset helper so the DTM stays pixel-aligned with the
    cropped image and mask (same geometry as ``center_crop``).
    """
    _, _, H, _ = dtm.shape
    top, left = compute_crop_offsets(H, crop_size)
    return dtm[:, :, top:top + crop_size, left:left + crop_size]


def train_one_epoch_gsl(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceFocalGSLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int,
    crop_size: int = CROP_SIZE,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: str = "fp16",
) -> Dict[str, float]:
    """Train one epoch with the scheduled Dice-Focal + GSL loss (paper Eq. 8).

    Mirrors :func:`train_one_epoch` (fp32 loss under AMP, gradient clipping on
    trainable params only) but feeds the per-class DTM to the criterion and sets
    the epoch-dependent ``alpha`` ONCE per epoch via ``criterion.set_epoch`` —
    so every batch in the epoch optimises the same blended objective, exactly as
    the paper intends (no per-batch change to the optimisation problem).

    The DataLoader must yield ``(image, mask, dtm)`` triples — i.e. it was built
    from ``BraTSDataset(..., return_dtm=True)``.

    Args:
        model:     Segmentation model in train mode.
        loader:    Training DataLoader yielding (image, mask, dtm).
        criterion: A :class:`DiceFocalGSLLoss` instance.
        optimizer: Optimiser (e.g. AdamW).
        device:    Computation device.
        epoch:     0-based epoch index — drives ``alpha`` via the scheduler.
        crop_size: Center-crop side length (applied to image, mask AND dtm).
        scaler:    Optional AMP GradScaler (fp16 only; ``None`` for bf16/none).
        amp_dtype: Autocast dtype: ``"fp16"``, ``"bf16"``, or ``"none"``.

    Returns:
        Dict of mean epoch metrics: ``loss``, ``region_loss``, ``gsl_loss``,
        ``alpha``, and ``dice_<class>`` × ``num_classes``.
    """
    model.train()
    tracker = MetricTracker()

    device_type = resolve_device_type(device)
    use_amp = amp_dtype != "none"
    autocast_dtype = resolve_amp_dtype(amp_dtype) if use_amp else None
    use_scaler = scaler is not None and scaler.is_enabled() and amp_dtype == "fp16"

    # Set alpha for this epoch ONCE (not per batch); the value is logged
    # per-batch via the alpha component returned by the criterion.
    criterion.set_epoch(epoch)

    for images, masks, dtms in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)   # [B, 4, H, W]
        masks  = masks.to(device,  non_blocking=True)   # [B, H, W]
        dtms   = dtms.to(device,   non_blocking=True)   # [B, C, H, W]

        images, masks = center_crop(images, masks, crop_size)
        dtms = _center_crop_dtm(dtms, crop_size)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=use_amp
        ):
            logits = model(images)                      # [B, C, H, W]

        # Loss in fp32 for numerical stability (region + GSL both sum over pixels).
        total, region, gsl, alpha_t = criterion(logits.float(), masks, dtms.float())

        if use_scaler:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
            optimizer.step()

        batch_metrics = {
            "loss":        total.item(),
            "region_loss": region.item(),
            "gsl_loss":    gsl.item(),
            "alpha":       float(alpha_t.item()),
        }
        batch_metrics.update(_compute_batch_dice(logits, masks))
        tracker.update(batch_metrics, n=images.size(0))

    return tracker.means()


@torch.no_grad()
def evaluate_gsl(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceFocalGSLLoss,
    device: torch.device | str,
    crop_size: int = CROP_SIZE,
) -> Dict[str, float]:
    """Evaluate with the scheduled Dice-Focal + GSL loss.

    Uses the criterion's CURRENT ``alpha`` (set by the most recent
    ``train_one_epoch_gsl`` call), so the reported validation loss is on the same
    blended objective as the matching training epoch. Runs at full precision
    under ``no_grad``.

    The DataLoader must yield ``(image, mask, dtm)`` triples.

    Returns:
        Dict of mean metrics: ``loss``, ``region_loss``, ``gsl_loss``,
        ``alpha``, and ``dice_<class>`` × ``num_classes``.
    """
    model.eval()
    tracker = MetricTracker()

    for images, masks, dtms in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        dtms   = dtms.to(device,   non_blocking=True)

        images, masks = center_crop(images, masks, crop_size)
        dtms = _center_crop_dtm(dtms, crop_size)

        logits = model(images)
        total, region, gsl, alpha_t = criterion(logits, masks, dtms)

        batch_metrics = {
            "loss":        total.item(),
            "region_loss": region.item(),
            "gsl_loss":    gsl.item(),
            "alpha":       float(alpha_t.item()),
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
