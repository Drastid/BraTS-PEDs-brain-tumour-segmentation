"""
src/dataset.py
==============
BraTS-PEDs 2D Slice Dataset — preprocessing utilities and PyTorch Dataset.

Preprocessing pipeline (applied per modality at load time):
    1. Load NIfTI → float32
    2. Clip outlier voxel intensities at 99.5th percentile (non-zero voxels)
       Rationale: scanner artifacts can produce spuriously high-intensity
       voxels that distort normalisation statistics.
    3. Z-score normalise on NON-ZERO voxels only.
       Rationale: background zeros are not brain tissue. Including them
       would drag μ toward 0 and suppress the actual brain tissue variance,
       making inter-modality comparisons meaningless.
    4. Force background (original zero voxels) back to 0 after normalisation.

Segmentation:
    - Labels: {0=BG, 1=NCR, 2=ED/SNFH, 3=ET}
    - Old BraTS convention uses label 4 for ET → remapped to 3 at load time.

Slice filtering:
    - Axial slices where brain coverage (non-zero T1n fraction) < 1% are
      discarded. These are top/bottom of the skull with no informative signal.
"""

import os
from typing import Callable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES: List[str] = ["t1c", "t1n", "t2f", "t2w"]
NUM_CLASSES: int = 4  # 0=BG, 1=NCR, 2=ED/SNFH, 3=ET


# ---------------------------------------------------------------------------
# Voxel-level preprocessing helpers
# ---------------------------------------------------------------------------


def clip_outliers(volume: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Clip non-zero voxel intensities to [0, p-th percentile].

    Eliminates scanner artifacts (spurious very-high-intensity voxels)
    before Z-score normalisation. Zero (background) voxels are unchanged.

    Args:
        volume:     3-D float32 array (H, W, D).
        percentile: Upper clipping percentile computed on non-zero voxels.

    Returns:
        Clipped float32 array of the same shape.
    """
    out = volume.copy().astype(np.float32)
    mask = out > 0
    if mask.sum() == 0:
        return out
    upper = float(np.percentile(out[mask], percentile))
    out[mask] = np.clip(out[mask], 0.0, upper)
    return out


def zscore_normalise(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise on non-zero (brain) voxels only.

    Background voxels (== 0 before normalisation) are forced back to 0
    after the transform, preserving the brain/background boundary.

    Args:
        volume: 3-D float32 array (H, W, D), already outlier-clipped.

    Returns:
        Normalised float32 array; background remains 0.
    """
    out = np.zeros_like(volume, dtype=np.float32)
    mask = volume > 0
    n = int(mask.sum())
    if n < 10:
        return out  # degenerate volume — return zeros
    mu = float(volume[mask].mean())
    sigma = float(volume[mask].std())
    if sigma < 1e-8:
        return out
    out[mask] = (volume[mask] - mu) / sigma
    return out


def count_outlier_voxels(
    volume: np.ndarray, sigma_threshold: float = 5.0
) -> Tuple[int, float]:
    """Quality-control helper: count non-zero voxels beyond ±N·σ.

    Returns:
        (n_outliers, fraction_of_brain_voxels)
    """
    mask = volume > 0
    if mask.sum() == 0:
        return 0, 0.0
    vals = volume[mask]
    mu, sigma = float(vals.mean()), float(vals.std())
    outliers = np.abs(vals - mu) > sigma_threshold * sigma
    n = int(outliers.sum())
    return n, n / len(vals)


# ---------------------------------------------------------------------------
# Subject-level I/O
# ---------------------------------------------------------------------------


def load_subject(
    subject_dir: str,
    subject_id: str,
    clip_percentile: float = 99.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess all modalities + segmentation for one subject.

    Args:
        subject_dir:     Path to the subject folder.
        subject_id:      Subject identifier string (e.g. 'BraTS-PED-00001-000').
        clip_percentile: Outlier clipping percentile (default 99.5).

    Returns:
        images: np.ndarray [4, H, W, D]  float32  (normalised, background=0)
        seg:    np.ndarray [H, W, D]     int8     (labels {0,1,2,3})
    """
    imgs: List[np.ndarray] = []
    for mod in MODALITIES:
        path = os.path.join(subject_dir, f"{subject_id}-{mod}.nii.gz")
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = clip_outliers(vol, percentile=clip_percentile)
        vol = zscore_normalise(vol)
        imgs.append(vol)
    images = np.stack(imgs, axis=0)  # [4, H, W, D]

    seg_path = os.path.join(subject_dir, f"{subject_id}-seg.nii.gz")
    seg = nib.load(seg_path).get_fdata().astype(np.int8)
    # Unify ET labels: old convention uses 4, new convention uses 3
    seg = np.where(seg == 4, 3, seg).astype(np.int8)

    return images, seg


# ---------------------------------------------------------------------------
# Slice filtering
# ---------------------------------------------------------------------------


def get_valid_slice_indices(
    subject_dir: str,
    subject_id: str,
    min_brain_fraction: float = 0.01,
) -> List[int]:
    """Return axial slice indices that contain sufficient brain tissue.

    Uses T1n as a cheap brain-presence proxy (only 1 volume loaded).
    A slice is kept if the fraction of non-zero T1n voxels ≥ min_brain_fraction.

    Args:
        subject_dir:        Path to the subject folder.
        subject_id:         Subject identifier string.
        min_brain_fraction: Minimum fraction of pixels with brain signal (default 1%).

    Returns:
        Sorted list of valid axial slice indices.
    """
    t1n_path = os.path.join(subject_dir, f"{subject_id}-t1n.nii.gz")
    t1n = nib.load(t1n_path).get_fdata()
    n_slices = t1n.shape[2]
    return [
        sl
        for sl in range(n_slices)
        if (t1n[:, :, sl] > 0).mean() >= min_brain_fraction
    ]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class BraTSDataset(Dataset):
    """Lightweight 2D slice dataset backed by pre-extracted .npy files.

    Each item is a (image, mask) pair:
        image : torch.Tensor [4, H, W]  float32  (4 normalised MRI channels)
        mask  : torch.Tensor [H, W]     int64    (class labels 0–3)

    Prerequisites
    -------------
    Run the ``extract_split`` function in ``02_preprocessing.ipynb`` first.
    It saves each valid axial slice as a .npy file under::

        data_dir/
            images/  <subject_id>_slice<idx>.npy   [4, H, W]  float32
            masks/   <subject_id>_slice<idx>.npy   [H, W]     int8

    Args:
        data_dir:  Path to the split directory (e.g. ``processed_dataset/train``).
        augment:   Optional albumentations Compose transform.
                   Must accept ``image`` (H,W,C float32) and
                   ``mask`` (H,W int8) keyword arguments.
    """

    def __init__(
        self,
        data_dir: str,
        augment: Optional[Callable] = None,
    ) -> None:
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        self.augment = augment

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.img_dir}\n"
                "Run extract_split() in 02_preprocessing.ipynb first."
            )

        self.files: List[str] = sorted(
            f for f in os.listdir(self.img_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        # Both reads are fast O(1) mmap-backed disk reads on pre-processed arrays.
        img: np.ndarray = np.load(os.path.join(self.img_dir, fname))  # [4, H, W] float32
        msk: np.ndarray = np.load(os.path.join(self.msk_dir, fname))  # [H, W]    int8

        if self.augment is not None:
            img_hwc = img.transpose(1, 2, 0)          # [H, W, 4]
            result  = self.augment(image=img_hwc, mask=msk)
            img     = result["image"].transpose(2, 0, 1)  # [4, H, W]
            msk     = result["mask"]

        return (
            torch.from_numpy(np.ascontiguousarray(img)),
            torch.from_numpy(np.ascontiguousarray(msk)).long(),
        )
