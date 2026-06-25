"""
src/constants.py
================
Shared constants for the BraTS-PEDs segmentation pipeline.

Centralising these values here keeps `dataset.py`, `train_utils.py`,
`eval_utils.py`, and `evaluate_3d_test.py` in sync. Any future
cross-domain dataset with different spatial dimensions only needs the
values in this file updated.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 4
CLASS_NAMES: Tuple[str, ...] = ("background", "NCR", "ED", "ET")

# ---------------------------------------------------------------------------
# Spatial dimensions (BraTS-PEDs defaults)
# ---------------------------------------------------------------------------

CROP_SIZE: int = 192   # center-crop side length used during training & evaluation
ORIG_SIZE: int = 240   # full H and W of each 2D .npy slice
N_SLICES: int = 155    # axial depth of every BraTS-PEDs volume

# ---------------------------------------------------------------------------
# Class-frequency vector (FULL dataset, 257 subjects = train+val+test).
# Background: 99.40 %   NCR: 0.066 %   ED: 0.441 %   ET: 0.091 %
# NOTE: used ONLY as a fallback. To avoid leaking val/test statistics into the
# loss, run_pipeline computes inverse-frequency class weights from the TRAIN
# split alone at runtime (_train_class_weights). These global numbers remain
# here for reference / offline use.
# ---------------------------------------------------------------------------

VOXEL_FREQ: np.ndarray = np.array(
    [0.9940, 0.00066, 0.00441, 0.00091],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Center-crop geometry (single source of truth)
# ---------------------------------------------------------------------------
#
# The deterministic center-crop offset is needed in two places that MUST agree
# exactly: training (`train_utils.center_crop`, applied to tensors) and 3D
# inference (`eval_utils.predict_volume`, applied to NumPy slices, with a
# matching un-crop). Previously the formula `(orig - crop) // 2` was copied in
# three locations; any divergence would silently shift predictions relative to
# the ground truth and corrupt every metric. Centralising it here guarantees
# train/eval consistency even if the crop strategy changes (e.g. A100 scaling).


def compute_crop_offsets(orig_size: int, crop_size: int) -> Tuple[int, int]:
    """Return the (top, left) offsets for a centered square crop.

    Args:
        orig_size: Side length of the source (square) canvas.
        crop_size: Side length of the desired square crop.

    Returns:
        ``(top, left)`` integer offsets such that the crop spans
        ``[top : top + crop_size, left : left + crop_size]``.

    Raises:
        ValueError: If ``crop_size`` exceeds ``orig_size``.
    """
    if crop_size > orig_size:
        raise ValueError(
            f"crop_size ({crop_size}) cannot exceed orig_size ({orig_size})."
        )
    top = (orig_size - crop_size) // 2
    left = (orig_size - crop_size) // 2
    return top, left
