"""
src/losses.py
=============
Custom loss functions for multi-class brain tumour segmentation.

Classes
-------
DiceLoss
    Soft multi-class Dice loss computed from logits.
FocalLoss
    Multi-class focal loss computed from logits.
CombinedLoss
    Weighted sum of DiceLoss + FocalLoss.

Design notes
------------
- All losses accept raw logits (no softmax/sigmoid applied externally).
- Targets are integer class indices of shape [B, H, W].
- Background (class 0) is included in the Dice average by default; it can be
  excluded via ``ignore_background=True`` to focus purely on tumour classes.
- Per-class weighting is supported in both losses to counteract the severe
  class imbalance present in BraTS (background ~99.4 % of all voxels).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss.

    Formula (per class c):

        Dice_c = 1 - (2 * Σ p_c * y_c + ε) / (Σ p_c + Σ y_c + ε)

    The final loss is the weighted mean across classes.

    Args:
        num_classes:       Number of segmentation classes (default 4).
        ignore_background: If True, class 0 is excluded from the mean.
                           Useful when background dominates and the model
                           trivially achieves near-1 Dice on it.
        class_weights:     Optional 1-D tensor of length ``num_classes``.
                           Weights are L1-normalised internally so they sum
                           to 1.  Ignored when ``ignore_background=True``
                           (per-class weights already reflect the focus on
                           foreground classes).
        smooth:            Laplace smoothing term ε to prevent division by
                           zero (default 1e-6).
    """

    def __init__(
        self,
        num_classes: int = 4,
        ignore_background: bool = True,
        class_weights: Optional[Sequence[float]] = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            w = w / w.sum()            # normalise to sum=1
            self.register_buffer("class_weights", w)
        else:
            self.class_weights: Optional[torch.Tensor] = None  # type: ignore[assignment]

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits:  Float tensor of shape [B, C, H, W] (raw, pre-softmax).
            targets: Long tensor  of shape [B, H, W]    (class indices).

        Returns:
            Scalar loss tensor.
        """
        # Softmax → class probabilities [B, C, H, W]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets → [B, C, H, W] float
        one_hot = (
            F.one_hot(targets, num_classes=self.num_classes)   # [B, H, W, C]
            .permute(0, 3, 1, 2)                               # [B, C, H, W]
            .float()
        )

        # Determine which classes to average over
        start_c = 1 if self.ignore_background else 0
        classes = list(range(start_c, self.num_classes))

        dice_per_class: list[torch.Tensor] = []
        for c in classes:
            p = probs[:, c]       # [B, H, W]
            y = one_hot[:, c]     # [B, H, W]
            intersection = (p * y).sum(dim=(1, 2))          # [B]
            cardinality  = p.sum(dim=(1, 2)) + y.sum(dim=(1, 2))  # [B]
            dice_c = 1.0 - (2.0 * intersection + self.smooth) / (
                cardinality + self.smooth
            )                                                # [B]
            dice_per_class.append(dice_c.mean())             # scalar

        dice_tensor = torch.stack(dice_per_class)            # [n_classes]

        if self.class_weights is not None and not self.ignore_background:
            w = self.class_weights[start_c:].to(dice_tensor.device)
            return (dice_tensor * w).sum()

        return dice_tensor.mean()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Focal loss downweights well-classified examples, forcing the network to
    concentrate on hard, misclassified examples — critical for the extreme
    class imbalance in BraTS (tumour voxels < 1 %).

    Formula:

        FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    where p_t is the model probability for the correct class.

    Args:
        gamma:        Focusing parameter γ ≥ 0.  γ=0 recovers standard
                      cross-entropy.  Typical values: 2.0–4.0 (default 2.0).
        alpha:        Optional per-class weight tensor (length C).
                      If None, uniform weights are used.
        ignore_index: Class index to ignore in the loss (e.g. -100 to skip
                      a ``void`` label). Default -100.
        reduction:    ``'mean'`` | ``'sum'`` | ``'none'`` (default ``'mean'``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            a = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a)
        else:
            self.alpha: Optional[torch.Tensor] = None  # type: ignore[assignment]

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:  Float tensor [B, C, H, W] (raw, pre-softmax).
            targets: Long tensor  [B, H, W]    (class indices).

        Returns:
            Scalar (or per-element) loss tensor.
        """
        # log-softmax is numerically more stable than log(softmax(x))
        log_probs = F.log_softmax(logits, dim=1)     # [B, C, H, W]
        probs     = log_probs.exp()                  # [B, C, H, W]

        # Reshape to [B*H*W, C] for gather operations
        B, C, H, W = logits.shape
        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
        probs_flat     = probs.permute(0, 2, 3, 1).reshape(-1, C)      # [N, C]
        targets_flat   = targets.reshape(-1)                            # [N]

        # Gather log-probabilities and probabilities for the true class
        log_pt = log_probs_flat.gather(1, targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)  # [N]
        pt     = probs_flat.gather(1,     targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)  # [N]

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma

        # Per-class α weight
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, targets_flat.clamp(min=0))             # [N]
            focal_weight = focal_weight * at

        # Raw focal loss per pixel
        loss = -focal_weight * log_pt                                   # [N]

        # Mask out ignored indices
        if self.ignore_index >= 0:
            valid = targets_flat != self.ignore_index
            loss  = loss[valid]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Weighted sum of DiceLoss and FocalLoss.

    Loss = dice_weight * DiceLoss + focal_weight * FocalLoss

    This combination is standard practice in medical image segmentation:
    - Dice loss directly optimises the overlap metric used for evaluation.
    - Focal loss provides pixel-wise gradient signal and handles class
      imbalance via the focusing parameter γ.

    Args:
        num_classes:       Number of segmentation classes (default 4).
        dice_weight:       Scalar weight for the Dice term (default 1.0).
        focal_weight:      Scalar weight for the Focal term (default 1.0).
        gamma:             Focal loss focusing parameter (default 2.0).
        ignore_background: Whether to exclude class 0 from Dice averaging
                           (default True — focus on tumour classes).
        class_weights:     Optional per-class weight sequence.  Applied to
                           both the Focal α and the Dice class weights.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        ignore_background: bool = True,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights if not ignore_background else None,
            smooth=1e-6,
        )
        self.focal_loss = FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            reduction="mean",
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss.

        Args:
            logits:  Float tensor [B, C, H, W] (raw, pre-softmax).
            targets: Long tensor  [B, H, W]    (class indices).

        Returns:
            total:  Weighted sum (scalar).
            d_loss: Raw Dice loss component (scalar, for logging).
            f_loss: Raw Focal loss component (scalar, for logging).
        """
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        total  = self.dice_weight * d_loss + self.focal_weight * f_loss
        return total, d_loss, f_loss
