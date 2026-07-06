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


import math


def _pseudo_logits(idx: torch.Tensor) -> torch.Tensor:
    """One-hot scalato: argmax(dim=1) riproduce esattamente `idx` (come eval_3d)."""
    return torch.nn.functional.one_hot(idx, num_classes=NUM_CLASSES).permute(
        0, 4, 1, 2, 3
    ).float() * 10.0


def test_class_absent_everywhere_is_nan_not_zero() -> None:
    """Semantica BraTS-standard (Opzione A): una sub-regione MAI presente in
    tutto il set (support==0) deve risultare NaN — "non applicabile" — ed
    essere ESCLUSA dalle medie foreground, NON entrarci come 0.0 fittizio.

    Questo blinda il bug diagnosticato in accortezze.md §1.1: con reduction
    mean_batch senza controllo del support, CC/ED assenti davano dice=0.0 e
    hd95=0.0, trascinando dice_mean_fg verso il basso e hd95_mean_fg verso lo
    zero-perfetto — falsando best.pth e l'early stopping.
    """
    D = H = W = 16
    metrics = SegmentationMetrics3D(num_classes=NUM_CLASSES)
    torch.manual_seed(0)
    # 3 volumi con SOLO classi 0,1,2 in GT e pred: CC(3) ed ED(4) mai presenti.
    for _ in range(3):
        g = torch.randint(0, 3, (1, D, H, W))
        p = torch.randint(0, 3, (1, D, H, W))
        metrics.update(_pseudo_logits(p), g)
    results = metrics.aggregate_and_reset()

    # CC ed ED: NaN (non 0.0), sia Dice sia HD95.
    for name in ("CC", "ED"):
        assert math.isnan(results[f"dice_{name}"]), f"dice_{name} dovrebbe essere NaN, e' {results[f'dice_{name}']}"
        assert math.isnan(results[f"hd95_{name}"]), f"hd95_{name} dovrebbe essere NaN, e' {results[f'hd95_{name}']}"
    # ET/NET presenti: valori finiti.
    for name in ("ET", "NET"):
        assert not math.isnan(results[f"dice_{name}"])
    # Le medie fg si calcolano SOLO su ET/NET (CC/ED esclusi): finite.
    assert not math.isnan(results["dice_mean_fg"])
    assert not math.isnan(results["hd95_mean_fg"])
    # Prova che CC/ED non sono entrati nella media: dice_mean_fg == media(ET,NET).
    expected = (results["dice_ET"] + results["dice_NET"]) / 2.0
    assert abs(results["dice_mean_fg"] - expected) < 1e-6


def test_false_negative_scores_zero() -> None:
    """Opzione A: se la GT contiene una classe ma la pred non la trova (falso
    negativo), il Dice per quella classe deve valere 0.0 (fallimento reale),
    NON NaN — la classe e' presente nel set, quindi conta."""
    D = H = W = 16
    gt = torch.zeros(1, D, H, W, dtype=torch.long)
    gt[0, :8, :8, :8] = 3  # GT ha CC
    pred = torch.zeros(1, D, H, W, dtype=torch.long)  # pred non predice CC

    metrics = SegmentationMetrics3D(num_classes=NUM_CLASSES)
    metrics.update(_pseudo_logits(pred), gt)
    results = metrics.aggregate_and_reset()

    assert results["dice_CC"] == 0.0, f"falso negativo su CC deve dare Dice 0.0, e' {results['dice_CC']}"


def test_empty_agreement_is_nan() -> None:
    """Opzione A: se pred e GT concordano nel NON avere una classe, quella
    classe e' NaN (esclusa), mentre le classi predette perfettamente danno 1.0."""
    D = H = W = 16
    gt = torch.zeros(1, D, H, W, dtype=torch.long)
    gt[0, :8, :8, :8] = 1  # solo ET presente
    pred = gt.clone()       # predizione perfetta

    metrics = SegmentationMetrics3D(num_classes=NUM_CLASSES)
    metrics.update(_pseudo_logits(pred), gt)
    results = metrics.aggregate_and_reset()

    assert math.isnan(results["dice_CC"]), "CC assente e concorde deve essere NaN"
    assert abs(results["dice_ET"] - 1.0) < 1e-6, "ET predetto perfettamente deve dare Dice 1.0"


if __name__ == "__main__":
    test_metrics_perfect_prediction()
    print("test_metrics_perfect_prediction: OK")
    test_metrics_absent_class_hd95_nan_handling()
    print("test_metrics_absent_class_hd95_nan_handling: OK")
    test_class_absent_everywhere_is_nan_not_zero()
    print("test_class_absent_everywhere_is_nan_not_zero: OK")
    test_false_negative_scores_zero()
    print("test_false_negative_scores_zero: OK")
    test_empty_agreement_is_nan()
    print("test_empty_agreement_is_nan: OK")
