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
# Class-frequency vector (from EDA on 257 subjects)
# Background: 99.40 %   NCR: 0.066 %   ED: 0.441 %   ET: 0.091 %
# ---------------------------------------------------------------------------

VOXEL_FREQ: np.ndarray = np.array(
    [0.9940, 0.00066, 0.00441, 0.00091],
    dtype=np.float32,
)
