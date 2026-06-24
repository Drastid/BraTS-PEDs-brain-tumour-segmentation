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
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from .constants import NUM_CLASSES

MODALITIES: List[str] = ["t1c", "t1n", "t2f", "t2w"]


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
# Distance Transform Map for the Generalized Surface Loss (Celaya et al.)
# ---------------------------------------------------------------------------


def compute_signed_dtm(
    mask_2d: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Per-class SIGNED Distance Transform Map of a 2D label mask (paper Fig. 2).

    For each class ``k`` the binary mask is converted to a signed Euclidean
    distance field ``D^k`` with the sign convention required by the Generalized
    Surface Loss:

        * **positive** outside the object,
        * **zero** on the boundary,
        * **negative** inside the object.

    Concretely, for the binary mask ``B`` of class ``k``::

        D = edt(~B)  -  edt(B)

    where ``edt`` is the Euclidean distance transform. ``edt(~B)`` gives the
    (positive) distance from each exterior voxel to the nearest object voxel;
    subtracting ``edt(B)`` makes the interior negative. Boundary voxels are 0 in
    both terms.

    Degenerate classes (absent in this slice) yield an all-zero ``D^k``: the EDT
    of an all-False mask is 0 everywhere, and ``~B`` is all-True whose EDT is
    also 0 (no zero-seed to measure distance to). An all-zero ``D^k`` makes that
    class contribute nothing to the GSL numerator and denominator — the correct
    behaviour (no surface to penalise).

    This is intended to be pre-computed OFFLINE per slice and stored alongside
    the image/mask ``.npy`` files (see ``scripts/precompute_gsl_stats.py``),
    avoiding the costly per-epoch DTM recomputation of the Hausdorff Loss
    (paper §1.1.2).

    Args:
        mask_2d:     Integer label array ``[H, W]`` with values in
                     ``{0, ..., num_classes-1}``.
        num_classes: Number of segmentation classes ``C``.

    Returns:
        Float32 array ``[C, H, W]`` — the signed DTM stacked per class.
    """
    edt = scipy.ndimage.distance_transform_edt
    h, w = mask_2d.shape
    dtm = np.zeros((num_classes, h, w), dtype=np.float32)

    for k in range(num_classes):
        binary = mask_2d == k
        if not binary.any():
            # Class absent in this slice → all-zero DTM (contributes nothing).
            continue
        if binary.all():
            # Class fills the whole slice (degenerate); no exterior → interior
            # distance only, kept negative for sign consistency.
            inside = edt(binary).astype(np.float32)
            dtm[k] = -inside
            continue
        outside = edt(~binary).astype(np.float32)   # >0 outside, 0 inside/boundary
        inside = edt(binary).astype(np.float32)     # >0 inside,  0 outside/boundary
        dtm[k] = outside - inside                   # +outside / -inside / 0 boundary

    return dtm


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
# Augmentation output normalisation (review.md §3.1)
# ---------------------------------------------------------------------------


def _to_chw_image(image) -> torch.Tensor:
    """Normalise an augmented image to a float32 CHW tensor [4, H, W].

    Handles both shapes albumentations can return:
      - ``np.ndarray`` in HWC layout (geometric-only Compose) → transpose to CHW.
      - ``torch.Tensor`` already in CHW layout (Compose ending in ToTensorV2).

    Args:
        image: Augmented image (NumPy HWC array or torch CHW tensor).

    Returns:
        Contiguous float32 tensor of shape [C, H, W].
    """
    if isinstance(image, torch.Tensor):
        return image.contiguous().float()
    # NumPy HWC → CHW
    chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


def _to_mask(mask) -> torch.Tensor:
    """Normalise an augmented mask to a long tensor [H, W] of class indices.

    Args:
        mask: Augmented mask (NumPy [H, W] array or torch tensor).

    Returns:
        Contiguous int64 tensor of shape [H, W].
    """
    if isinstance(mask, torch.Tensor):
        return mask.contiguous().long()
    return torch.from_numpy(np.ascontiguousarray(mask)).long()


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
        data_dir:   Path to the split directory (e.g. ``processed_dataset/train``).
        augment:    Optional albumentations Compose transform.
                    Must accept ``image`` (H,W,C float32) and
                    ``mask`` (H,W int8) keyword arguments. When ``return_dtm`` is
                    True the transform must also accept a ``dtm`` additional
                    target of type ``image`` (see ``get_augmentation``).
        return_dtm: If True, also load and return the per-class signed Distance
                    Transform Map for the Generalized Surface Loss. Requires the
                    DTMs to have been pre-computed (see
                    ``scripts/precompute_gsl_stats.py``). ``__getitem__`` then
                    returns a 3-tuple ``(image, mask, dtm)``; otherwise it
                    returns the original 2-tuple ``(image, mask)`` for backward
                    compatibility.
        dtm_dir:    Override for the DTM directory. Defaults to
                    ``<data_dir>/dtms``.
    """

    def __init__(
        self,
        data_dir: str,
        augment: Optional[Callable] = None,
        return_dtm: bool = False,
        dtm_dir: Optional[str] = None,
    ) -> None:
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        self.augment = augment
        self.return_dtm = return_dtm
        self.dtm_dir = dtm_dir if dtm_dir is not None else os.path.join(data_dir, "dtms")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.img_dir}\n"
                "Run extract_split() in 02_preprocessing.ipynb first."
            )

        if self.return_dtm and not os.path.isdir(self.dtm_dir):
            raise FileNotFoundError(
                f"DTM directory not found: {self.dtm_dir}\n"
                "Run `python -m scripts.precompute_gsl_stats` first to generate "
                "the per-slice Distance Transform Maps required by the GSL."
            )

        self.files: List[str] = sorted(
            f for f in os.listdir(self.img_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        # Each read loads a small pre-processed slice array into RAM (fast: the
        # heavy NIfTI decoding happened once, offline, during preprocessing).
        img: np.ndarray = np.load(os.path.join(self.img_dir, fname))  # [4, H, W] float32
        msk: np.ndarray = np.load(os.path.join(self.msk_dir, fname))  # [H, W]    int8

        dtm: Optional[np.ndarray] = None
        if self.return_dtm:
            dtm = np.load(os.path.join(self.dtm_dir, fname))          # [C, H, W] float32

        if self.augment is not None:
            # albumentations expects channels-last (H, W, C) for the image.
            img_hwc = np.transpose(img, (1, 2, 0))    # [H, W, 4]
            if dtm is not None:
                # The DTM is passed as an additional image-type target so it
                # undergoes the SAME spatial transform as image/mask, keeping it
                # aligned with the augmented mask. The DTM-aware augmentation
                # pipeline (get_augmentation(with_dtm=True)) is restricted to
                # rigid isometries (flip / rot90), under which the transformed
                # DTM remains an EXACT distance map — no interpolation, so the
                # GSL gradient is not polluted.
                dtm_hwc = np.transpose(dtm, (1, 2, 0))   # [H, W, C]
                result = self.augment(image=img_hwc, mask=msk, dtm=dtm_hwc)
                dtm = result["dtm"]
            else:
                result = self.augment(image=img_hwc, mask=msk)
            # The output type depends on the pipeline (review.md §3.1):
            #   - geometric-only Compose → NumPy arrays (current default)
            #   - Compose ending in ToTensorV2 → torch tensors (already CHW img)
            # _to_chw_image / _to_mask normalise both cases to a stable contract.
            img = result["image"]
            msk = result["mask"]
            if dtm is not None:
                return _to_chw_image(img), _to_mask(msk), _to_chw_image(dtm)
            return _to_chw_image(img), _to_mask(msk)

        # No augmentation: img is already [4, H, W], msk is [H, W].
        img_t = torch.from_numpy(np.ascontiguousarray(img)).float()
        msk_t = torch.from_numpy(np.ascontiguousarray(msk)).long()
        if dtm is not None:
            dtm_t = torch.from_numpy(np.ascontiguousarray(dtm)).float()
            return img_t, msk_t, dtm_t
        return img_t, msk_t
