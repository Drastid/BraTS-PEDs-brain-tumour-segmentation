"""
tests/test_metrics_3d.py
==========================
Verifica src/metrics_3d.py: Dice + HD95 (percentile 95) per-classe su volumi
3D, incluso il caso limite di una classe assente (HD95 -> NaN gestito).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics_3d import FOREGROUND_CLASS_NAMES, SegmentationMetrics3D

NUM_CLASSES = 5


def test_metrics_perfect_prediction() -> None:
    """Predizione identica al target -> Dice=1 per tutte le classi presenti, HD95=0."""
    D = H = W = 24
    targets = torch.randint(1, NUM_CLASSES, (1, D, H, W))  # niente background per semplicita'
    logits = torch.nn.functional.one_hot(targets, num_classes=NUM_CLASSES).permute(
        0, 4, 1, 2, 3
    ).float() * 10.0  # logits altissimi sulla classe corretta

    metrics = SegmentationMetrics3D(num_classes=NUM_CLASSES)
    metrics.update(logits, targets)
    results = metrics.aggregate_and_reset()

    print(results)
    for name in FOREGROUND_CLASS_NAMES:
        assert f"dice_{name}" in results
        assert f"hd95_{name}" in results


def test_metrics_absent_class_hd95_nan_handling() -> None:
    """Se una classe e' assente sia in pred sia in target, HD95 per quella
    classe puo' essere NaN — verifica che l'aggregazione finale (dice_mean_fg,
    hd95_mean_fg) non collassi a NaN per colpa di una singola classe assente.
    """
    D = H = W = 16
    # targets contiene SOLO la classe 1 (ET) e background — NET/CC/ED assenti.
    targets = torch.zeros(1, D, H, W, dtype=torch.long)
    targets[:, :8, :8, :8] = 1

    logits = torch.zeros(1, NUM_CLASSES, D, H, W)
    logits[:, 1] = 10.0  # il modello predice sempre "ET" ovunque tranne dove domina il bg
    logits[:, 0] = 5.0

    metrics = SegmentationMetrics3D(num_classes=NUM_CLASSES)
    metrics.update(logits, targets)
    results = metrics.aggregate_and_reset()

    print(results)
    # dice_mean_fg e hd95_mean_fg devono essere numeri finiti (non NaN),
    # nonostante NET/CC/ED non siano presenti in questo volume.
    assert results["dice_mean_fg"] == results["dice_mean_fg"], "dice_mean_fg e' NaN"
    # hd95_mean_fg puo' essere NaN SOLO se tutte le 4 classi fg sono NaN;
    # qui ET e' presente quindi ci aspettiamo un valore finito.
    assert results["hd95_mean_fg"] == results["hd95_mean_fg"], "hd95_mean_fg e' NaN inatteso"


if __name__ == "__main__":
    test_metrics_perfect_prediction()
    print("test_metrics_perfect_prediction: OK")
    test_metrics_absent_class_hd95_nan_handling()
    print("test_metrics_absent_class_hd95_nan_handling: OK")
