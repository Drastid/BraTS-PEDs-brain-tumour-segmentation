"""
src/config.py
=============
Centralised training configuration for the BraTS-PEDs segmentation pipeline.

Rationale
---------
Before this module existed, every training notebook (``03_train_unet``,
``05_train_segformer``, ``06_train_fpn``) carried its own hand-copied block of
hyper-parameters (batch size, learning rates, epoch counts, loss weights, …).
Keeping three identical copies in sync was error-prone: a single forgotten edit
silently desynchronised one model from the others.

This file is now the **single source of truth**.  Notebooks import from here::

    from src.config import TrainConfig
    cfg = TrainConfig()
    loader = DataLoader(ds, batch_size=cfg.batch_size, ...)

Spatial constants (``NUM_CLASSES``, ``CROP_SIZE``, ``ORIG_SIZE`` …) are NOT
redefined here — they live in :mod:`src.constants` and are re-exported so callers
have one import surface.  ``TrainConfig`` pulls its ``crop_size`` default from
``constants.CROP_SIZE`` to avoid a third copy of that number.

The dataclass is intentionally plain (no YAML/JSON layer) to keep the notebooks
dependency-free; it is trivially serialisable via :meth:`TrainConfig.to_dict`
for logging into a run registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

from .constants import (
    CLASS_NAMES,     # noqa: F401 — re-exported
    CROP_SIZE,
    NUM_CLASSES,
    ORIG_SIZE,       # noqa: F401 — re-exported
)


@dataclass
class TrainConfig:
    """Hyper-parameters shared by all three model-training notebooks.

    Every field has the exact default value that was previously hard-coded in
    the notebooks, so importing ``TrainConfig()`` reproduces the original
    behaviour bit-for-bit.  Override individual fields to run experiments::

        cfg = TrainConfig(batch_size=64, crop_size=224)
    """

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Model / architecture ─────────────────────────────────────────────
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    segformer_checkpoint: str = "nvidia/mit-b1"
    num_classes: int = NUM_CLASSES
    in_channels: int = 4              # t1c, t1n, t2f, t2w
    crop_size: int = CROP_SIZE        # pulled from constants — single source

    # ── Data loading ─────────────────────────────────────────────────────
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # ── Two-phase schedule ───────────────────────────────────────────────
    phase1_epochs: int = 5            # encoder frozen
    phase2_epochs: int = 25           # full fine-tuning

    lr_phase1: float = 3e-4           # decoder-only LR (higher)
    lr_phase2: float = 1e-4           # full-model LR (lower)
    encoder_lr_mult: float = 0.1      # encoder LR = lr_phase2 * this in Phase 2
    weight_decay: float = 1e-4

    # ── Early stopping (anti-overfitting) ────────────────────────────────
    # When enabled, Phase 2 terminates if the monitored val foreground Dice
    # does not improve by > es_min_delta for `es_patience` consecutive epochs.
    # The monitor / best-checkpoint selection stays GLOBAL across both phases
    # (see run_pipeline); only the *termination* is confined to Phase 2, since
    # Phase 1 is a fixed short warm-up where val is still rising.
    early_stopping: bool = False
    es_patience: int = 6              # epochs without improvement before stop
    es_min_delta: float = 0.0         # minimum val fg-Dice gain to count as "better"

    # ── Loss ─────────────────────────────────────────────────────────────
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_gamma: float = 2.0
    ignore_background: bool = True    # exclude class-0 from Dice average

    # ── Augmentation ─────────────────────────────────────────────────────
    augment_prob: float = 0.5
    # Scales the *magnitude* of the geometric transforms (shift/scale/rotate
    # limits and elastic alpha). 1.0 = legacy behaviour; >1.0 = stronger.
    augment_strength: float = 1.0
    # If True, also add Z-score-safe intensity augmentations (additive Gaussian
    # noise + mild multiplicative brightness/contrast) on top of the geometric
    # set. Off by default to preserve the exact legacy pipeline.
    augment_intensity: bool = False

    # ── Mixed precision ──────────────────────────────────────────────────
    # "fp16" matches the legacy GradScaler behaviour; "bf16" is recommended on
    # Ampere+ (A100) for better numerical stability (see review.md §6.2).
    amp_dtype: str = "fp16"          # one of {"fp16", "bf16", "none"}

    # ── Derived / convenience ────────────────────────────────────────────
    @property
    def total_epochs(self) -> int:
        """Total number of training epochs across both phases."""
        return self.phase1_epochs + self.phase2_epochs

    def to_dict(self) -> Dict:
        """Flat, JSON-serialisable view of the config (for run logging)."""
        d = asdict(self)
        d["total_epochs"] = self.total_epochs
        return d


# Convenience module-level singleton: notebooks can do
#   ``from src.config import CONFIG`` and read fields directly, or instantiate
#   their own ``TrainConfig(...)`` for ablations.
CONFIG: TrainConfig = TrainConfig()
