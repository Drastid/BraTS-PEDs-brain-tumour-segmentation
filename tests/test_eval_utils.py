"""Unit tests for src/eval_utils.py — 3D metrics and post-processing."""

import warnings

import numpy as np

from src.eval_utils import (
    compute_dice_volume,
    compute_hd95_volume,
    compute_iou_volume,
    remove_small_components,
)


def test_dice_perfect_volume():
    """Identical volumes → per-class Dice = 1.0 for every class."""
    np.random.seed(0)
    vol = np.random.randint(0, 4, size=(10, 10, 10), dtype=np.int64)
    for key, value in compute_dice_volume(vol, vol).items():
        assert abs(value - 1.0) < 1e-5, f"{key} = {value}"


def test_iou_perfect_volume():
    """Identical volumes → per-class IoU = 1.0 for every class."""
    np.random.seed(0)
    vol = np.random.randint(0, 4, size=(10, 10, 10), dtype=np.int64)
    for key, value in compute_iou_volume(vol, vol).items():
        assert abs(value - 1.0) < 1e-5, f"{key} = {value}"


def test_remove_small_components():
    """Blobs of size 10/60/1000 → only the 60- and 1000-voxel ones survive min_voxels=50."""
    vol = np.zeros((40, 40, 40), dtype=np.int64)
    vol[0:2, 0:5, 0:1]       = 1   # 10 voxels — will be removed
    vol[10:13, 10:14, 10:15] = 1   # 60 voxels — will survive
    vol[20:30, 20:30, 20:30] = 1   # 1000 voxels — will survive

    out = remove_small_components(vol, min_voxels=50)

    assert int(out[0:2, 0:5, 0:1].sum()) == 0
    assert int(out[10:13, 10:14, 10:15].sum()) == 60
    assert int(out[20:30, 20:30, 20:30].sum()) == 1000


def test_hd95_edge_cases():
    """Both-empty / GT-only / pred-only → NaN; normal case → positive float."""
    pred = np.zeros((8, 8, 8), dtype=np.int64)
    gt   = np.zeros((8, 8, 8), dtype=np.int64)

    # Case A — both empty
    assert np.isnan(compute_hd95_volume(pred, gt)["hd95_NCR"])

    # Cases B and C emit RuntimeWarning by design — silence them in the test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        gt_only = gt.copy(); gt_only[1, 1, 1] = 1
        assert np.isnan(compute_hd95_volume(pred, gt_only)["hd95_NCR"])

        pred_only = pred.copy(); pred_only[1, 1, 1] = 1
        assert np.isnan(compute_hd95_volume(pred_only, gt)["hd95_NCR"])

    # Normal — separated single voxels → finite positive value
    p, g = pred.copy(), gt.copy()
    p[0, 0, 0] = 1
    g[7, 7, 7] = 1
    h = compute_hd95_volume(p, g)["hd95_NCR"]
    assert np.isfinite(h) and h > 0, f"Expected positive HD95, got {h}"
