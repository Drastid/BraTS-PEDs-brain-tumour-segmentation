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
                           Weights are L1-normalised internally. They are
                           applied to whichever classes are averaged: when
                           ``ignore_background=True`` the background weight is
                           dropped and the remaining foreground weights are
                           re-normalised to sum to 1.
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

        # Per-class weighting (review.md §5.1, §5.3).
        # Previously this branch required ``not self.ignore_background``, so with
        # the production setting ``ignore_background=True`` the Dice weights were
        # silently dropped — the inverse-frequency weights only affected the
        # Focal term. We now apply the weights to whichever classes are actually
        # averaged (``start_c:``) and RE-NORMALISE that subset to sum to 1, so
        # the weighted mean stays on the same scale as the unweighted one.
        if self.class_weights is not None:
            w = self.class_weights[start_c:].to(dice_tensor.device)
            w = w / w.sum()
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

        # Pass class_weights to the Dice term UNCONDITIONALLY (review.md §5.1).
        # DiceLoss now applies them to the averaged (foreground) classes and
        # re-normalises, so the weights are no longer silently dropped when
        # ignore_background=True — they act on both Dice and Focal consistently.
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights,
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


# ===========================================================================
# Generalized Surface Loss (GSL) — Celaya et al., arXiv:2302.03868
# ===========================================================================
#
# This block implements the boundary-aware training objective described in the
# paper, adapted to this project's 2D slice-based pipeline:
#
#   L(t) = alpha(t) * L_DiceFocal  +  (1 - alpha(t)) * L_GSL              (Eq. 8)
#
# Components:
#   - compute_global_class_weights : pre-computed global weights w_k       (Eq. 13)
#   - AlphaScheduler               : linear / step / cosine schedules  (Eq. 14-16)
#   - GeneralizedSurfaceLoss       : the GSL term itself                  (Eq. 12)
#   - DiceFocalGSLLoss             : the full dynamically-scheduled loss   (Eq. 8)
#
# Key design choices faithful to the paper:
#   * w_k are GLOBAL (computed once over the whole dataset) and injected, NOT
#     recomputed per batch — this is exactly what distinguishes the GSL from the
#     Generalized Dice Loss and prevents per-batch gradient oscillation
#     (paper §1.1.1 and §2.1, requirement reproduced in the project brief).
#   * The signed ground-truth Distance Transform Map (D) is pre-computed offline
#     and fed in per sample, avoiding the per-epoch DTM recomputation that makes
#     the Hausdorff Loss expensive (paper §1.1.2).
#   * The GSL is bounded in [0, 1] by construction (Eq. 11-12), so it shares the
#     scale of the region loss in the convex combination (paper §2.1).


def compute_global_class_weights(
    voxel_counts: Sequence[float],
    eps: float = 1e-12,
) -> "torch.Tensor":
    """Pre-compute the GSL global class weights ``w_k`` (paper Eq. 13).

    .. math::
        w_k = \\left( \\frac{1}{\\sum_{j} 1/N_j} \\right) \\frac{1}{N_k}

    where ``N_k`` is the TOTAL number of voxels of class ``k`` across the WHOLE
    dataset. Because the weights are derived from dataset-level counts they are
    constant throughout training; injecting them (rather than recomputing per
    batch, as the Generalized Dice Loss does) keeps the optimisation problem
    fixed and avoids gradient oscillation (paper §1.1.1, §2.1).

    The result is L1-normalised so the weights sum to 1 — this does not change
    the GSL value (it appears in both numerator and denominator) but keeps the
    weight vector on a stable, interpretable scale for logging.

    Args:
        voxel_counts: Per-class total voxel counts ``[N_0, ..., N_{C-1}]`` over
                      the entire dataset (typically produced offline by
                      ``scripts/precompute_gsl_stats`` / the preprocessing step).
        eps:          Small constant to guard against division by zero for a
                      class that is absent from the dataset.

    Returns:
        Float tensor of shape ``[C]`` with the normalised global weights.
    """
    counts = torch.as_tensor(voxel_counts, dtype=torch.float64)
    inv = 1.0 / (counts + eps)               # 1 / N_k
    w = inv / inv.sum()                       # (1 / N_k) / sum_j (1 / N_j)  == Eq. 13 normalised
    return w.to(dtype=torch.float32)


class AlphaScheduler:
    """Epoch-indexed schedule for the GSL mixing weight ``alpha(t)`` (Eq. 14-16).

    ``alpha`` starts at 1 (region loss only) and decreases to 0 in the final
    epoch (GSL only). Three schedules from the paper are supported:

    - ``"linear"`` (Eq. 14):  ``alpha(t) = 1 - t / T``
    - ``"step"``   (Eq. 15):  ``alpha(t) = 1 - floor(t/h) / floor(T/h)``
    - ``"cosine"`` (Eq. 16):  ``alpha(t) = 0.5 * (1 + cos(pi * t / T))``

    The step schedule holds ``alpha`` constant for ``h`` epochs at a time, which
    lets an optimiser such as AdamW settle on each sub-objective before the
    gradient target shifts (paper §4 — this is the rationale the project brief
    asks for).

    Convention for ``t``:
        ``t`` is the 0-based epoch index, ``t in {0, ..., T-1}``, and ``T`` is the
        total number of epochs. With this convention ``alpha(0) = 1`` for all
        schedules, and the linear / step schedules reach ``alpha = 0`` at the
        final epoch ``t = T - 1``. (The cosine schedule approaches but only hits
        exactly 0 at ``t = T``; over ``[0, T-1]`` it ends just above 0, matching
        the paper's "decrease until alpha = 0 in the final epoch" intent.)

    Args:
        schedule:     One of ``"linear"``, ``"step"``, ``"cosine"``.
        total_epochs: ``T`` — total number of training epochs (both phases).
        step_length:  ``h`` — step length for the step schedule (ignored
                      otherwise). Must be >= 1.

    Raises:
        ValueError: For an unknown schedule name or ``step_length < 1``.
    """

    def __init__(
        self,
        schedule: str = "step",
        total_epochs: int = 30,
        step_length: int = 5,
    ) -> None:
        if schedule not in ("linear", "step", "cosine"):
            raise ValueError(
                f"Unknown alpha schedule {schedule!r}; expected 'linear', "
                f"'step', or 'cosine'."
            )
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}.")
        if step_length < 1:
            raise ValueError(f"step_length must be >= 1, got {step_length}.")
        self.schedule = schedule
        self.total_epochs = int(total_epochs)
        self.step_length = int(step_length)

    def __call__(self, epoch: int) -> float:
        """Return ``alpha`` for a 0-based ``epoch`` index, clamped to [0, 1]."""
        import math

        t = int(epoch)
        T = self.total_epochs
        # Denominator uses T-1 so that the final epoch (t = T-1) maps to alpha=0
        # for the linear/step schedules, matching the paper's intent over the
        # [0, T-1] training range.
        denom = max(T - 1, 1)

        if self.schedule == "linear":
            alpha = 1.0 - t / denom
        elif self.schedule == "step":
            h = self.step_length
            n_h = max((T - 1) // h, 1)        # number of steps over [0, T-1]
            alpha = 1.0 - (t // h) / n_h
        else:  # cosine
            alpha = 0.5 * (1.0 + math.cos(math.pi * t / denom))

        return float(min(1.0, max(0.0, alpha)))


class GeneralizedSurfaceLoss(nn.Module):
    """Generalized Surface Loss — boundary-aware, bounded in [0, 1] (Eq. 12).

    .. math::
        L_{gsl} = 1 - \\frac{\\sum_k w_k \\sum_i \\big(D_i^k (1 - (T_i^k + P_i^k))\\big)^2}
                              {\\sum_k w_k \\sum_i (D_i^k)^2}

    where:
        * ``P_i^k`` is the softmax probability of class ``k`` at voxel ``i``,
        * ``T_i^k`` is the one-hot ground truth,
        * ``D_i^k`` is the SIGNED ground-truth Distance Transform Map of class
          ``k`` (positive outside the object, zero on the boundary, negative
          inside — paper Fig. 2),
        * ``w_k`` are the pre-computed global class weights (Eq. 13).

    Numerical properties (paper §2.1): the denominator is the "worst-case"
    response ``sum (D_i^k)^2`` (independent of the prediction), so the ratio lies
    in ``[0, 1]`` and the loss is bounded — unlike the Boundary Loss. This keeps
    its magnitude comparable to the region loss in the convex combination.

    Inputs are raw logits; softmax is applied internally so the term composes
    with the rest of the pipeline exactly like ``DiceLoss`` / ``FocalLoss``.

    Args:
        num_classes:       Number of segmentation classes ``C``.
        class_weights:     Global weights ``w_k`` of length ``C`` (from
                           :func:`compute_global_class_weights`). If ``None``,
                           uniform weights are used (not recommended for the
                           imbalanced BraTS setting). Registered as a buffer so
                           it follows ``.to(device)``.
        include_background: Whether class 0 contributes to the GSL sums. The
                           paper sums over all classes; default ``True`` keeps
                           that behaviour. Set ``False`` to focus purely on
                           foreground surfaces (consistent with
                           ``ignore_background`` elsewhere).
        eps:               Small constant stabilising the normalisation when a
                           class is absent from a batch (then both numerator and
                           denominator contributions for that class are ~0).
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[Sequence[float]] = None,
        include_background: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() != num_classes:
                raise ValueError(
                    f"class_weights must have length num_classes={num_classes}, "
                    f"got {w.numel()}."
                )
        else:
            w = torch.ones(num_classes, dtype=torch.float32) / num_classes
        self.register_buffer("class_weights", w)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dtm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the GSL.

        Args:
            logits:  Float tensor ``[B, C, H, W]`` (raw, pre-softmax).
            targets: Long tensor  ``[B, H, W]``    (class indices).
            dtm:     Float tensor ``[B, C, H, W]`` — the SIGNED ground-truth
                     Distance Transform Map per class, pre-computed offline and
                     supplied by the DataLoader. Must already be center-cropped
                     to match ``logits`` spatially.

        Returns:
            Scalar GSL value in ``[0, 1]``.
        """
        probs = F.softmax(logits, dim=1)                       # P: [B, C, H, W]
        one_hot = (
            F.one_hot(targets, num_classes=self.num_classes)   # [B, H, W, C]
            .permute(0, 3, 1, 2)
            .to(probs.dtype)                                   # T: [B, C, H, W]
        )

        start_c = 0 if self.include_background else 1
        w = self.class_weights[start_c:].to(probs.device)      # [C']
        D = dtm[:, start_c:]                                    # [B, C', H, W]
        T = one_hot[:, start_c:]
        P = probs[:, start_c:]

        # Numerator term per voxel:  ( D * (1 - (T + P)) )^2          (Eq. 12)
        num_vox = (D * (1.0 - (T + P))) ** 2                    # [B, C', H, W]
        # Denominator term per voxel: D^2  (the prediction-independent worst case)
        den_vox = D ** 2                                        # [B, C', H, W]

        # Sum over voxels (H, W), then weight per class and sum over classes.
        num_per_class = num_vox.sum(dim=(0, 2, 3))             # [C']
        den_per_class = den_vox.sum(dim=(0, 2, 3))             # [C']

        numerator = (w * num_per_class).sum()
        denominator = (w * den_per_class).sum()

        return 1.0 - numerator / (denominator + self.eps)


class DiceFocalGSLLoss(nn.Module):
    """Dynamically-scheduled Dice-Focal + Generalized Surface Loss (Eq. 8).

    .. math::
        L(t) = \\alpha(t)\\, L_{DiceFocal} + (1 - \\alpha(t))\\, L_{GSL}

    At the start of training ``alpha = 1`` so the objective is the familiar
    region loss (Dice + Focal); as epochs advance ``alpha`` decreases toward 0
    and the boundary-aware GSL progressively takes over, directly targeting the
    Hausdorff-based metrics (paper §2, §3).

    The schedule is owned by an :class:`AlphaScheduler`. ``alpha`` is updated
    once per epoch via :meth:`set_epoch` (called by the training loop), NOT per
    batch — so every batch within an epoch optimises the same blended objective.

    Args:
        num_classes:       Number of segmentation classes.
        gsl_class_weights: Global weights ``w_k`` for the GSL (Eq. 13).
        scheduler:         An :class:`AlphaScheduler`. If ``None``, a step
                           schedule (``h=5``) over 30 epochs is created — the
                           project's default.
        dice_weight:       Weight of the Dice term inside the region loss.
        focal_weight:      Weight of the Focal term inside the region loss.
        gamma:             Focal focusing parameter.
        ignore_background: Region-loss background handling (see
                           :class:`CombinedLoss`).
        region_class_weights: Per-class weights for the region (Dice/Focal)
                           term. May differ from the GSL weights; defaults to
                           ``None`` (uniform), matching prior CombinedLoss usage
                           unless explicitly set.
        gsl_include_background: Whether the GSL sums over the background class.
    """

    def __init__(
        self,
        num_classes: int = 4,
        gsl_class_weights: Optional[Sequence[float]] = None,
        scheduler: Optional[AlphaScheduler] = None,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        ignore_background: bool = True,
        region_class_weights: Optional[Sequence[float]] = None,
        gsl_include_background: bool = True,
    ) -> None:
        super().__init__()
        self.region_loss = CombinedLoss(
            num_classes=num_classes,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            gamma=gamma,
            ignore_background=ignore_background,
            class_weights=region_class_weights,
        )
        self.gsl = GeneralizedSurfaceLoss(
            num_classes=num_classes,
            class_weights=gsl_class_weights,
            include_background=gsl_include_background,
        )
        self.scheduler = scheduler if scheduler is not None else AlphaScheduler(
            schedule="step", total_epochs=30, step_length=5
        )
        # alpha for the current epoch; defaults to the schedule value at epoch 0
        # (== 1.0 → region loss only) until set_epoch() is called.
        self._alpha: float = self.scheduler(0)

    def set_epoch(self, epoch: int) -> float:
        """Update ``alpha`` for ``epoch`` (0-based). Call once per epoch.

        Returns the new ``alpha`` so the training loop can log it.
        """
        self._alpha = self.scheduler(epoch)
        return self._alpha

    @property
    def alpha(self) -> float:
        """Current GSL mixing weight ``alpha(t)``."""
        return self._alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dtm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the scheduled combined loss.

        Args:
            logits:  Float tensor ``[B, C, H, W]`` (raw logits).
            targets: Long tensor  ``[B, H, W]``.
            dtm:     Float tensor ``[B, C, H, W]`` — signed GT DTM (cropped).

        Returns:
            total:    ``alpha * region + (1 - alpha) * gsl`` (scalar).
            region:   The Dice+Focal region loss (scalar, for logging).
            gsl:      The GSL term (scalar, for logging).
            alpha:    The current ``alpha`` as a tensor (for logging).
        """
        region, _d, _f = self.region_loss(logits, targets)
        gsl = self.gsl(logits, targets, dtm)
        a = self._alpha
        total = a * region + (1.0 - a) * gsl
        alpha_t = torch.as_tensor(a, dtype=total.dtype, device=total.device)
        return total, region, gsl, alpha_t
