"""
tests/test_transforms_3d.py
=============================
Verifica src/transforms_3d.py::ClipAndNormalizeNonZerod contro
un'implementazione di riferimento in NumPy che replica ESATTAMENTE la logica
del vecchio progetto 2D (master/src/dataset.py::clip_outliers +
zscore_normalise), per dimostrare che il porting a torch/MONAI e' numericamente
equivalente (a meno di errori di floating point).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transforms_3d import ClipAndNormalizeNonZerod


def _reference_clip_and_normalize(volume: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Riferimento NumPy — stessa logica esatta del vecchio
    master/src/dataset.py (clip_outliers poi zscore_normalise), fusa in un
    'unica funzione per un confronto diretto.
    """
    out = volume.copy().astype(np.float32)
    mask = out > 0
    if mask.sum() == 0:
        return np.zeros_like(out)

    upper = float(np.percentile(out[mask], percentile))
    clipped = np.clip(out[mask], 0.0, upper)

    mu = float(clipped.mean())
    sigma = float(clipped.std())

    result = np.zeros_like(out)
    if sigma < 1e-8:
        return result
    result[mask] = (clipped - mu) / sigma
    return result


def test_matches_reference_numpy_implementation() -> None:
    rng = np.random.default_rng(42)
    D, H, W = 20, 24, 28
    C = 4

    # Simula un volume MRI: background=0 in una regione, intensita' positive
    # altrove, con qualche outlier estremo (scanner artifact) da clippare.
    volume_np = np.zeros((C, D, H, W), dtype=np.float32)
    for c in range(C):
        brain_mask = rng.random((D, H, W)) > 0.3  # ~70% "brain" tissue
        vals = rng.normal(loc=500, scale=100, size=(D, H, W)).astype(np.float32)
        vals = np.clip(vals, 1, None)  # valori positivi
        # Inietta outlier estremi in pochi voxel (simula artefatto scanner)
        outlier_mask = rng.random((D, H, W)) > 0.999
        vals[outlier_mask] = 50000.0
        volume_np[c] = np.where(brain_mask, vals, 0.0)

    expected = np.stack(
        [_reference_clip_and_normalize(volume_np[c]) for c in range(C)], axis=0
    )

    transform = ClipAndNormalizeNonZerod(keys=["image"])
    result = transform({"image": torch.from_numpy(volume_np)})
    actual = result["image"].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_background_stays_exactly_zero() -> None:
    D = H = W = 16
    volume = torch.zeros(4, D, H, W)
    volume[:, :8, :8, :8] = torch.rand(4, 8, 8, 8) * 1000 + 1  # brain region positiva
    # Il resto resta 0 (background).

    transform = ClipAndNormalizeNonZerod(keys=["image"])
    result = transform({"image": volume})
    out = result["image"]

    background_mask = volume == 0
    assert torch.all(out[background_mask] == 0.0), "Il background deve restare esattamente 0"


def test_degenerate_channel_returns_zeros() -> None:
    """Canale con troppi pochi voxel non-zero (< min_nonzero_voxels) -> tutto zero."""
    volume = torch.zeros(1, 10, 10, 10)
    volume[0, 0, 0, 0] = 500.0  # un solo voxel non-zero, sotto la soglia di 10

    transform = ClipAndNormalizeNonZerod(keys=["image"], min_nonzero_voxels=10)
    result = transform({"image": volume})

    assert torch.all(result["image"] == 0.0)


if __name__ == "__main__":
    test_matches_reference_numpy_implementation()
    print("test_matches_reference_numpy_implementation: OK")
    test_background_stays_exactly_zero()
    print("test_background_stays_exactly_zero: OK")
    test_degenerate_channel_returns_zeros()
    print("test_degenerate_channel_returns_zeros: OK")
