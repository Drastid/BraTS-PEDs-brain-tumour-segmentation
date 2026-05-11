"""Unit tests for src/dataset.py — voxel-level preprocessing helpers."""

import numpy as np

from src.dataset import clip_outliers, zscore_normalise


def test_clip_outliers():
    """A single 1e6 outlier on top of U[0, 100] is clipped to ~100 at p99.5."""
    np.random.seed(0)
    vol = np.random.uniform(0, 100, size=(20, 20, 20)).astype(np.float32)
    vol[0, 0, 0] = 1e6
    clipped = clip_outliers(vol, percentile=99.5)
    assert clipped.max() < 110, f"max after clip = {clipped.max()}"
    assert clipped[1, 1, 1] == vol[1, 1, 1]   # non-outlier voxel unchanged


def test_zscore_non_zero_only():
    """Background zeros stay 0; non-zero voxels are rescaled to μ≈0 σ≈1."""
    np.random.seed(0)
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros_like(vol, dtype=bool)
    mask[:5] = True   # first half is non-zero
    vol[mask] = np.random.uniform(10, 20, size=int(mask.sum())).astype(np.float32)

    out = zscore_normalise(vol)

    assert (out[~mask] == 0).all(), "background voxels must remain 0"
    fg = out[mask]
    assert abs(fg.mean()) < 1e-5, f"foreground μ = {fg.mean()}"
    assert abs(fg.std() - 1.0) < 1e-5, f"foreground σ = {fg.std()}"
