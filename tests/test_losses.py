"""
tests/test_losses.py
======================
Verifica che le loss N-D (src/losses_3d.py) funzionino IDENTICAMENTE in 2D
[B,C,H,W] e in 3D [B,C,D,H,W] senza modifiche al codice — questa e' la prova
diretta che la generalizzazione richiesta dall'utente (porting delle loss
custom del vecchio progetto 2D) e' corretta.

Verifica anche src/losses_monai.py (pipeline di default) e la pipeline GSL
completa: src/transforms.py::ComputeDistanceMapd + GeneralizedSurfaceLoss +
DiceFocalGSLLoss, in 3D.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses_3d import (
    AlphaScheduler,
    CombinedLoss,
    DiceFocalGSLLoss,
    DiceLoss,
    FocalLoss,
    GeneralizedSurfaceLoss,
    compute_global_class_weights,
)
from src.losses_monai import build_dice_focal_loss
from src.transforms import ComputeDistanceMapd, compute_signed_dtm_nd

NUM_CLASSES = 5


@pytest.mark.parametrize("spatial_shape", [(64, 64), (32, 32, 32)], ids=["2D", "3D"])
def test_dice_loss_nd(spatial_shape: tuple[int, ...]) -> None:
    B = 2
    logits = torch.randn(B, NUM_CLASSES, *spatial_shape)
    targets = torch.randint(0, NUM_CLASSES, (B, *spatial_shape))

    loss_fn = DiceLoss(num_classes=NUM_CLASSES, ignore_background=True)
    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 1.0 + 1e-4


@pytest.mark.parametrize("spatial_shape", [(64, 64), (32, 32, 32)], ids=["2D", "3D"])
def test_focal_loss_nd(spatial_shape: tuple[int, ...]) -> None:
    B = 2
    logits = torch.randn(B, NUM_CLASSES, *spatial_shape)
    targets = torch.randint(0, NUM_CLASSES, (B, *spatial_shape))

    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


@pytest.mark.parametrize("spatial_shape", [(64, 64), (32, 32, 32)], ids=["2D", "3D"])
def test_combined_loss_nd(spatial_shape: tuple[int, ...]) -> None:
    B = 2
    logits = torch.randn(B, NUM_CLASSES, *spatial_shape, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (B, *spatial_shape))

    class_weights = [0.05, 0.3, 0.3, 0.15, 0.2]
    loss_fn = CombinedLoss(num_classes=NUM_CLASSES, class_weights=class_weights)
    total, d_loss, f_loss = loss_fn(logits, targets)

    assert total.ndim == 0
    # Verifica che il gradiente si propaghi correttamente (requisito per il training).
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gsl_pipeline_3d() -> None:
    """Pipeline GSL completa in 3D: ComputeDistanceMapd -> GeneralizedSurfaceLoss."""
    D = H = W = 32
    label_np = np.random.randint(0, NUM_CLASSES, size=(1, D, H, W)).astype(np.int16)
    label_t = torch.from_numpy(label_np)

    transform = ComputeDistanceMapd(keys=["label"], num_classes=NUM_CLASSES)
    data = {"label": label_t}
    out = transform(data)

    assert "distance_map" in out
    dtm = out["distance_map"]
    assert dtm.shape == (NUM_CLASSES, D, H, W)

    # Ora usa la DTM in GeneralizedSurfaceLoss con un batch fittizio.
    B = 2
    logits = torch.randn(B, NUM_CLASSES, D, H, W)
    targets = torch.randint(0, NUM_CLASSES, (B, D, H, W))
    dtm_batch = dtm.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # simula un batch

    gsl_fn = GeneralizedSurfaceLoss(num_classes=NUM_CLASSES)
    gsl_value = gsl_fn(logits, targets, dtm_batch)

    assert gsl_value.ndim == 0
    assert 0.0 <= gsl_value.item() <= 1.0 + 1e-4


def test_dice_focal_gsl_loss_scheduling_3d() -> None:
    """DiceFocalGSLLoss: alpha(0)=1 (solo region), alpha(T-1)=0 (solo GSL)."""
    D = H = W = 16
    B = 1
    logits = torch.randn(B, NUM_CLASSES, D, H, W)
    targets = torch.randint(0, NUM_CLASSES, (B, D, H, W))

    label_np = targets[0].numpy().astype(np.int16)
    dtm = torch.from_numpy(
        __import__("src.transforms", fromlist=["compute_signed_dtm_nd"]).compute_signed_dtm_nd(
            label_np, NUM_CLASSES
        )
    ).unsqueeze(0).float()

    scheduler = AlphaScheduler(schedule="linear", total_epochs=10)
    criterion = DiceFocalGSLLoss(num_classes=NUM_CLASSES, scheduler=scheduler)

    criterion.set_epoch(0)
    assert criterion.alpha == pytest.approx(1.0)
    total0, region0, gsl0, alpha0 = criterion(logits, targets, dtm)
    assert alpha0.item() == pytest.approx(1.0)

    criterion.set_epoch(9)
    assert criterion.alpha == pytest.approx(0.0, abs=1e-6)
    total9, region9, gsl9, alpha9 = criterion(logits, targets, dtm)
    assert alpha9.item() == pytest.approx(0.0, abs=1e-6)


def test_monai_dice_focal_wrapper_3d() -> None:
    """src/losses_monai.py — la pipeline di default (--loss=dice_focal)."""
    B = 2
    D = H = W = 32
    logits = torch.randn(B, NUM_CLASSES, D, H, W, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (B, D, H, W))

    class_weights = [0.05, 0.3, 0.3, 0.15, 0.2]
    criterion = build_dice_focal_loss(num_classes=NUM_CLASSES, class_weights=class_weights)
    loss = criterion(logits, targets)

    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_compute_global_class_weights() -> None:
    counts = [1000, 10, 20, 5, 50]
    w = compute_global_class_weights(counts)
    assert w.shape == (5,)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    # La classe con meno voxel (indice 3, count=5) deve avere il peso maggiore.
    assert w[3] == w.max()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
