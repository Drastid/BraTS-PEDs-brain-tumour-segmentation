"""Unit tests for src/losses.py — DiceLoss and FocalLoss."""

import torch
import torch.nn.functional as F

from src.losses import DiceLoss, FocalLoss


def test_dice_loss_perfect_prediction():
    """Perfect one-hot predictions on every voxel → DiceLoss ≈ 0."""
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 8, 8
    targets = torch.randint(0, C, (B, H, W))
    logits = torch.full((B, C, H, W), -1e3)
    logits.scatter_(1, targets.unsqueeze(1), 1e3)
    loss = DiceLoss(num_classes=C, ignore_background=False)(logits, targets)
    assert loss.item() < 1e-3, f"Expected ~0, got {loss.item()}"


def test_dice_loss_inverted():
    """Predicting the wrong class everywhere → DiceLoss > 0.5 (gradient direction sanity)."""
    B, C, H, W = 2, 4, 8, 8
    targets = torch.full((B, H, W), 1, dtype=torch.long)
    logits = torch.full((B, C, H, W), -1e3)
    logits[:, 2, :, :] = 1e3
    loss = DiceLoss(num_classes=C, ignore_background=True)(logits, targets)
    assert loss.item() > 0.5, f"Expected > 0.5, got {loss.item()}"


def test_focal_loss_zero_gamma_equals_ce():
    """γ=0 and α=None → FocalLoss reduces to cross-entropy."""
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 8, 8
    logits = torch.randn((B, C, H, W))
    targets = torch.randint(0, C, (B, H, W))
    fl = FocalLoss(gamma=0.0, alpha=None, ignore_index=-100, reduction="mean")(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(fl, ce, atol=1e-6), f"FL={fl.item()} vs CE={ce.item()}"
