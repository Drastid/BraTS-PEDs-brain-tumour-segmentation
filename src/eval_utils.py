"""
src/eval_utils.py
=================
Clinical evaluation utilities for the BraTS-PEDs segmentation pipeline.

This module bridges the 2D slice-level model with 3D volumetric clinical
metrics.  Functions are intentionally stateless and composable so they can
be called from any evaluation notebook or script.

Public API (Phase 4)
--------------------
SUB-TASK 1 — 3D Reconstruction & Post-Processing
    get_subject_ids()           → List[str]       (all subjects in a split dir)
    infer_dataset_shape()       → (int, int)       (auto-detect orig_size, n_slices)
    predict_volume()            → (pred, gt)       (3D inference for one subject)
    load_subject_nifti_meta()   → nib.Nifti1Image  (affine + header from NIfTI)
    volume_to_nifti()           → nib.Nifti1Image  (wrap array as NIfTI)
    remove_small_components()   → np.ndarray       (post-process: kill confetti)

SUB-TASK 2 — Clinical Metrics (HD95)
    compute_dice_volume()       → Dict[str, float] (per-class Dice on 3D volumes)
    compute_hd95_volume()       → Dict[str, float] (per-class HD95 on 3D volumes)

Constants
---------
CROP_SIZE  : 192   – center-crop side length (must match training configuration)
ORIG_SIZE  : 240   – full spatial dimension of NIfTI and .npy slices (BraTS default)
N_SLICES   : 155   – axial depth of BraTS-PEDs volumes (dataset-specific default)
CLASS_NAMES: tuple – ("background", "NCR", "ED", "ET")

Cross-domain usage
------------------
All functions that accept ``data_dir`` work on any dataset whose preprocessed
``.npy`` slices follow the ``<subject_id>_slice<idx>.npy`` naming convention
(produced by ``02_preprocessing.ipynb``).  Use ``infer_dataset_shape()`` to
auto-detect ``orig_size`` and ``n_slices`` instead of relying on the
BraTS-PEDs defaults.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn

from .constants import CLASS_NAMES, CROP_SIZE, NUM_CLASSES, N_SLICES, ORIG_SIZE

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Pre-compute crop offsets once (same formula as train_utils.center_crop)
_CROP_TOP: int = (ORIG_SIZE - CROP_SIZE) // 2    # 24
_CROP_LEFT: int = (ORIG_SIZE - CROP_SIZE) // 2   # 24


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _parse_slice_idx(filename: str) -> int:
    """Extract the integer slice index from ``<subject_id>_slice<idx>.npy``.

    Args:
        filename: Basename of the .npy file (e.g. ``BraTS-PED-00004-000_slice007.npy``).

    Returns:
        Zero-based integer slice index.

    Raises:
        ValueError: If the expected ``_slice<digits>.npy`` suffix is absent.
    """
    match = re.search(r"_slice(\d+)\.npy$", filename)
    if match is None:
        raise ValueError(f"Cannot parse slice index from filename: {filename!r}")
    return int(match.group(1))


def _subject_id_from_filename(filename: str) -> str:
    """Return the subject-ID prefix from ``<subject_id>_slice<idx>.npy``.

    Args:
        filename: Basename of the .npy file.

    Returns:
        Subject identifier string (e.g. ``BraTS-PED-00004-000``).
    """
    return filename.rsplit("_slice", 1)[0]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def infer_dataset_shape(data_dir: str, subject_id: str) -> Tuple[int, int]:
    """Auto-detect the spatial dimensions of a preprocessed dataset split.

    Inspects the ``.npy`` slice files for one subject to determine:
    - ``orig_size``: the H (= W) dimension of each 2D slice array.
    - ``n_slices``:  the total axial depth (number of slices) for the subject.

    This replaces the hardcoded BraTS-PEDs defaults (``ORIG_SIZE=240``,
    ``N_SLICES=155``) when evaluating on a new dataset whose dimensions may
    differ.  Use the returned values as ``orig_size`` and ``n_slices`` in
    :func:`predict_volume`.

    Args:
        data_dir:   Path to the split root (e.g. ``data/processed_adult_brats/test``).
        subject_id: Any valid subject identifier present in the split.

    Returns:
        ``(orig_size, n_slices)`` — a 2-tuple of ints.

    Raises:
        FileNotFoundError: If no slice files are found for ``subject_id``.
        ValueError:        If slice arrays have unexpected dimensionality.
    """
    img_dir = os.path.join(data_dir, "images")
    slice_files: List[str] = sorted(
        [f for f in os.listdir(img_dir) if f.startswith(subject_id + "_slice")],
        key=_parse_slice_idx,
    )
    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for subject {subject_id!r} in {img_dir!r}."
        )

    # Load one file to detect the spatial dimension
    sample: np.ndarray = np.load(os.path.join(img_dir, slice_files[0]))
    if sample.ndim != 3:  # expected [C, H, W]
        raise ValueError(
            f"Expected 3-D array [C, H, W] but got shape {sample.shape} "
            f"in {slice_files[0]!r}."
        )
    _, h, w = sample.shape
    if h != w:
        raise ValueError(
            f"Non-square slice ({h}×{w}) found in {slice_files[0]!r}. "
            "Only square spatial dims are supported by center_crop."
        )
    orig_size: int = h

    # Detect the highest slice index to determine axial depth
    max_idx: int = max(_parse_slice_idx(f) for f in slice_files)
    n_slices: int = max_idx + 1  # indices are 0-based

    return orig_size, n_slices


def get_subject_ids(data_dir: str) -> List[str]:
    """Return a sorted list of unique subject IDs present in a split directory.

    Subject IDs are inferred from the basenames of ``.npy`` files in the
    ``images/`` sub-directory, following the ``<subject_id>_slice<idx>.npy``
    naming convention produced by ``02_preprocessing.ipynb``.

    Args:
        data_dir: Path to the split root (e.g. ``processed_dataset/val``).

    Returns:
        Sorted list of unique subject-ID strings.

    Raises:
        FileNotFoundError: If the ``images/`` sub-directory does not exist.
    """
    img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    ids = {
        _subject_id_from_filename(f)
        for f in os.listdir(img_dir)
        if f.endswith(".npy")
    }
    return sorted(ids)


# ---------------------------------------------------------------------------
# 3D inference (slice-by-slice)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_volume(
    model: nn.Module,
    data_dir: str,
    subject_id: str,
    device: torch.device | str,
    crop_size: int = CROP_SIZE,
    orig_size: Optional[int] = None,
    n_slices: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 2D slice-by-slice inference and reconstruct a 3D label volume.

    Each pre-processed ``.npy`` slice is loaded, the same ``crop_size × crop_size``
    center crop that was applied during training is re-applied, the model produces
    a per-pixel class prediction, and the result is un-cropped back into the
    ``orig_size × orig_size`` spatial canvas.  The per-slice predictions are
    stacked in the original axial order (axis 2) to form the 3D output.

    Axial slices that were discarded during preprocessing (brain coverage < 1 %)
    are kept at their background-zero default — in practice this dataset has
    no filtered-out slices (all 155 slices pass the 1 % threshold).

    **Cross-domain note**: ``orig_size`` and ``n_slices`` default to ``None``,
    which triggers automatic detection via :func:`infer_dataset_shape`.  Pass
    explicit values to override (e.g. when you already know the dimensions and
    want to avoid the extra file-system scan).

    Args:
        model:      Segmentation model.  **Must already be in ``eval`` mode**
                    and transferred to ``device`` before calling this function.
        data_dir:   Path to the split root directory
                    (e.g. ``processed_dataset/val`` or
                    ``data/processed_adult_brats/test``).
        subject_id: Subject identifier (e.g. ``BraTS-PED-00004-000``).
        device:     Torch device for inference tensors.
        crop_size:  Center-crop side length used at training time (default 192).
        orig_size:  Full spatial dimension of .npy slices.  ``None`` (default)
                    triggers auto-detection from the slice files.
        n_slices:   Total axial depth of the volume.  ``None`` (default)
                    triggers auto-detection from the slice files.

    Returns:
        pred_vol : np.ndarray shape ``[orig_size, orig_size, n_slices]`` int8.
                   Predicted class labels {0, 1, 2, 3}.
        gt_vol   : np.ndarray shape ``[orig_size, orig_size, n_slices]`` int8.
                   Ground-truth class labels loaded from pre-processed ``.npy``
                   mask files.

    Raises:
        FileNotFoundError: If no slice files are found for ``subject_id``.
    """
    img_dir = os.path.join(data_dir, "images")
    msk_dir = os.path.join(data_dir, "masks")

    # Collect and sort all slice files belonging to this subject
    slice_files: List[str] = sorted(
        [f for f in os.listdir(img_dir) if f.startswith(subject_id + "_slice")],
        key=_parse_slice_idx,
    )
    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for subject {subject_id!r} in {img_dir!r}."
        )

    # Auto-detect spatial dimensions when not provided by the caller.
    # This makes predict_volume() work on any dataset without hardcoded dims.
    if orig_size is None or n_slices is None:
        _detected_orig, _detected_n = infer_dataset_shape(data_dir, subject_id)
        if orig_size is None:
            orig_size = _detected_orig
        if n_slices is None:
            n_slices = _detected_n

    # Pre-compute center-crop offsets (same formula as train_utils.center_crop)
    top  = (orig_size - crop_size) // 2
    left = (orig_size - crop_size) // 2

    # Allocate volumes: unvisited slices default to background (label 0)
    pred_vol = np.zeros((orig_size, orig_size, n_slices), dtype=np.int8)
    gt_vol   = np.zeros((orig_size, orig_size, n_slices), dtype=np.int8)

    model.eval()
    for fname in slice_files:
        sl_idx = _parse_slice_idx(fname)

        # Load pre-processed slice arrays
        img: np.ndarray = np.load(os.path.join(img_dir, fname))   # [4, H, W] float32
        msk: np.ndarray = np.load(os.path.join(msk_dir, fname))   # [H, W]    int8

        # Apply the same center crop used during training
        img_crop = img[:, top:top + crop_size, left:left + crop_size]  # [4, 192, 192]

        # Forward pass → argmax prediction [192, 192]
        img_tensor = torch.from_numpy(img_crop).unsqueeze(0).to(device)  # [1,4,192,192]
        logits     = model(img_tensor)                                     # [1,4,192,192]
        pred_crop  = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)

        # Un-crop: write prediction into the center region of the 240×240 canvas
        pred_vol[top:top + crop_size, left:left + crop_size, sl_idx] = pred_crop

        # Ground-truth: store the full 240×240 mask
        gt_vol[:, :, sl_idx] = msk

    return pred_vol, gt_vol


# ---------------------------------------------------------------------------
# NIfTI I/O helpers
# ---------------------------------------------------------------------------


def load_subject_nifti_meta(
    nifti_root: str,
    subject_id: str,
    seg_filename: Optional[str] = None,
) -> nib.Nifti1Image:
    """Load the segmentation NIfTI image to retrieve its affine and header.

    The affine encodes the voxel-to-world-space transform (patient orientation,
    voxel spacing, origin) and is needed to create a valid NIfTI output that
    radiologists and ITK tools can read correctly.

    Args:
        nifti_root:    Path to the directory containing per-subject sub-folders.
                       E.g. ``PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training``.
        subject_id:    Subject identifier (e.g. ``BraTS-PED-00004-000``).
        seg_filename:  Optional override for the segmentation filename inside
                       the subject folder.  Defaults to
                       ``"{subject_id}-seg.nii.gz"`` (BraTS convention).  Set
                       this when the target dataset uses a different naming
                       scheme (e.g. ``"seg.nii.gz"`` or ``"label.nii.gz"``).

    Returns:
        ``nibabel.Nifti1Image`` of the segmentation file.
        Access ``img.affine`` and ``img.header`` as needed.

    Raises:
        FileNotFoundError: If the segmentation file does not exist.
    """
    fname = seg_filename if seg_filename is not None else f"{subject_id}-seg.nii.gz"
    seg_path = os.path.join(nifti_root, subject_id, fname)
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation NIfTI not found: {seg_path!r}")
    return nib.load(seg_path)


def volume_to_nifti(
    volume: np.ndarray,
    affine: np.ndarray,
    header: Optional[nib.Nifti1Header] = None,
) -> nib.Nifti1Image:
    """Wrap a 3D integer label array as a NIfTI image with the original affine.

    Args:
        volume: np.ndarray ``[H, W, D]`` integer labels.
        affine: 4×4 float64 affine matrix from the reference NIfTI.
        header: Optional NIfTI header copied from the reference image.
                Preserves metadata such as voxel dimensions and data type.

    Returns:
        ``nibabel.Nifti1Image`` ready to be saved with ``nibabel.save()``.
    """
    return nib.Nifti1Image(volume.astype(np.int16), affine, header)


# ---------------------------------------------------------------------------
# Post-processing: remove small isolated components
# ---------------------------------------------------------------------------


def remove_small_components(
    volume: np.ndarray,
    min_voxels: int = 50,
) -> np.ndarray:
    """Remove isolated 3D connected components smaller than ``min_voxels``.

    The "confetti effect" — scattered single-voxel or small-island predictions
    far from the main tumour mass — is a common failure mode of 2D slice-based
    models.  This function eliminates such artifacts by processing each
    non-background label class independently:

    1. Extract a binary mask for the class.
    2. Compute 3D connected components with 26-connectivity.
    3. Relabel any component with fewer than ``min_voxels`` voxels to 0 (BG).

    Only non-background classes (NCR=1, ED=2, ET=3) are filtered.

    Args:
        volume:     np.ndarray ``[H, W, D]`` integer labels {0, 1, 2, 3}.
        min_voxels: Minimum component size to retain (default 50 voxels).
                    Components strictly smaller than this value are removed.

    Returns:
        Post-processed label volume of the same shape and dtype as ``volume``.
    """
    out = volume.copy()
    # 26-connectivity: a voxel is connected to its 26 face/edge/corner neighbours
    struct = scipy.ndimage.generate_binary_structure(3, 3)

    for cls in range(1, NUM_CLASSES):   # skip class 0 (background)
        binary_mask = volume == cls
        if not binary_mask.any():
            continue

        labeled_array, n_components = scipy.ndimage.label(binary_mask, structure=struct)

        for comp_id in range(1, n_components + 1):
            component_mask = labeled_array == comp_id
            if int(component_mask.sum()) < min_voxels:
                out[component_mask] = 0   # erase small component to background

    return out


# ---------------------------------------------------------------------------
# Volumetric Dice (SUB-TASK 2 prerequisite — also used by HD95 pipeline)
# ---------------------------------------------------------------------------


def compute_dice_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-class Dice coefficient on 3D label volumes.

    Uses the standard formula:
        Dice_c = (2 · |P_c ∩ G_c| + ε) / (|P_c| + |G_c| + ε)

    Args:
        pred:   np.ndarray ``[H, W, D]`` integer predicted labels.
        gt:     np.ndarray ``[H, W, D]`` integer ground-truth labels.
        smooth: Laplace smoothing term (default 1e-6) to avoid division by zero
                when both prediction and ground-truth are empty for a class.
                Returns 1.0 (perfect agreement) in that degenerate case.

    Returns:
        Dict mapping ``"dice_<classname>"`` → float in [0, 1].
        Includes all 4 classes: background, NCR, ED, ET.
    """
    results: Dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        pred_c = (pred == c).astype(np.float32)
        gt_c   = (gt   == c).astype(np.float32)
        inter  = float((pred_c * gt_c).sum())
        denom  = float(pred_c.sum() + gt_c.sum())
        dice   = (2.0 * inter + smooth) / (denom + smooth)
        results[f"dice_{name}"] = dice
    return results


def compute_iou_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-class Intersection-over-Union (Jaccard) on 3D label volumes.

    Uses the standard formula:
        IoU_c = (|P_c ∩ G_c| + ε) / (|P_c ∪ G_c| + ε)

    Args:
        pred:   np.ndarray ``[H, W, D]`` integer predicted labels.
        gt:     np.ndarray ``[H, W, D]`` integer ground-truth labels.
        smooth: Laplace smoothing term (default 1e-6) to avoid division by zero
                when both prediction and ground-truth are empty for a class.
                Returns 1.0 (perfect agreement) in that degenerate case.

    Returns:
        Dict mapping ``"iou_<classname>"`` → float in [0, 1].
        Includes all 4 classes: background, NCR, ED, ET.
    """
    results: Dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        pred_c = (pred == c)
        gt_c   = (gt   == c)
        inter  = float((pred_c & gt_c).sum())
        union  = float((pred_c | gt_c).sum())
        results[f"iou_{name}"] = (inter + smooth) / (union + smooth)
    return results


# ---------------------------------------------------------------------------
# HD95 (SUB-TASK 2)
# ---------------------------------------------------------------------------


def compute_hd95_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute the 95th-percentile Hausdorff Distance (HD95) per tumour class.

    Uses ``medpy.metric.binary.hd95``, the reference implementation adopted by
    the BraTS challenge organisers.  HD95 measures the worst-case surface-to-
    surface distance between prediction and ground-truth contours at the 95th
    percentile, expressed in physical units (mm for BraTS, since voxel spacing
    is 1 mm isotropic).

    Only foreground classes are evaluated — HD95 on the background class is
    clinically meaningless and is therefore omitted.

    Edge-case handling (CRITICAL)
    -----------------------------
    ``medpy.metric.binary.hd95`` requires at least one non-zero voxel in **both**
    ``result`` and ``reference``.  Three distinct degenerate situations arise:

    Case A — both pred and GT empty for a class
        The class is not present in either volume.  There is nothing to measure.
        Convention: return ``np.nan``.  The downstream aggregation code must use
        ``np.nanmean`` to exclude these cases from the per-class mean.

    Case B — GT has class, pred is entirely empty (false-negative miss)
        The model failed to detect the tumour sub-region at all.
        Convention: return ``np.nan`` and emit a ``WARNING`` message.
        Rationale: assigning a single scalar penalty (e.g. the diagonal of the
        volume) would be arbitrary and would distort ranking; instead, NaN is
        propagated so the caller can count "complete misses" separately.

    Case C — pred has class, GT is entirely empty (false-positive hallucination)
        The model predicts a class that does not exist in the ground truth.
        Convention: return ``np.nan`` and emit a ``WARNING`` message.
        Rationale: same as Case B — no GT surface exists to compute distance to.

    Any other unexpected exception from medpy is caught, logged, and returned
    as ``np.nan`` to prevent the evaluation loop from aborting mid-run.

    Args:
        pred:          np.ndarray ``[H, W, D]`` integer predicted labels {0,1,2,3}.
        gt:            np.ndarray ``[H, W, D]`` integer ground-truth labels {0,1,2,3}.
        voxel_spacing: Physical voxel size in mm per axis (H, W, D).
                       BraTS-PEDs voxels are 1 mm isotropic → default (1.0, 1.0, 1.0).

    Returns:
        Dict mapping ``"hd95_<classname>"`` → float (mm) or ``np.nan``.
        Keys: ``"hd95_NCR"``, ``"hd95_ED"``, ``"hd95_ET"``
        (background is excluded).
    """
    from medpy.metric.binary import hd95 as _medpy_hd95

    results: Dict[str, float] = {}

    # Evaluate only foreground classes (skip index 0 = background)
    for c in range(1, NUM_CLASSES):
        name = CLASS_NAMES[c]
        key  = f"hd95_{name}"

        pred_bin: np.ndarray = (pred == c)
        gt_bin:   np.ndarray = (gt   == c)

        pred_any: bool = bool(pred_bin.any())
        gt_any:   bool = bool(gt_bin.any())

        # --- Case A: both empty — class absent in both volumes ---
        if not pred_any and not gt_any:
            results[key] = float("nan")
            continue

        # --- Case B: GT present, prediction entirely missing ---
        if not pred_any and gt_any:
            warnings.warn(
                f"[HD95] Class {name!r}: GT has {int(gt_bin.sum())} voxels "
                f"but prediction is completely empty (complete miss). "
                f"HD95 set to NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            results[key] = float("nan")
            continue

        # --- Case C: prediction present, GT entirely empty ---
        if pred_any and not gt_any:
            warnings.warn(
                f"[HD95] Class {name!r}: prediction has {int(pred_bin.sum())} voxels "
                f"but GT is completely empty (false-positive hallucination). "
                f"HD95 set to NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            results[key] = float("nan")
            continue

        # --- Normal case: both pred and GT are non-empty ---
        try:
            value = _medpy_hd95(
                pred_bin,
                gt_bin,
                voxelspacing=voxel_spacing,
                connectivity=1,
            )
            results[key] = float(value)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[HD95] Class {name!r}: medpy raised an unexpected exception "
                f"({type(exc).__name__}: {exc}). HD95 set to NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            results[key] = float("nan")

    return results
