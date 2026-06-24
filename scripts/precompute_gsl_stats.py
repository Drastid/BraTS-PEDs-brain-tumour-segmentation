"""
scripts/precompute_gsl_stats.py
===============================
Offline pre-computation for the Generalized Surface Loss (Celaya et al.).

Two products, both computed ONCE before training so nothing is recomputed
per-epoch (the per-epoch DTM recomputation is precisely what makes the
Hausdorff Loss expensive — paper §1.1.2):

1. **Global class weights w_k (Eq. 13).**
   Counts the total number of voxels of each class over the TRAIN split and
   derives ``w_k = (1 / sum_j 1/N_j) * (1 / N_k)``. These dataset-level weights
   are constant during training, so injecting them keeps the optimisation
   problem fixed and avoids the per-batch gradient oscillation of the
   Generalized Dice Loss (paper §1.1.1, §2.1). Saved as JSON.

2. **Per-slice signed Distance Transform Maps (DTM).**
   For every ``.npy`` mask in each split, computes the per-class signed DTM
   (positive outside / negative inside / zero on the boundary — paper Fig. 2)
   and stores it as ``dtms/<same_filename>.npy`` with shape ``[C, H, W]``
   float32, mirroring the ``images/`` and ``masks/`` layout so the DataLoader
   can load it by filename.

Usage
-----
    python -m scripts.precompute_gsl_stats                  # all splits, default root
    python -m scripts.precompute_gsl_stats --weights-only   # only recompute w_k
    python -m scripts.precompute_gsl_stats --splits train val
    python -m scripts.precompute_gsl_stats --archive tar    # also pack DTMs per split
    python -m scripts.precompute_gsl_stats --archive zip --archive-dir /content/gsl

Outputs
-------
    processed_dataset/gsl_class_weights.json
    processed_dataset/<split>/dtms/<subject>_slice<idx>.npy   [C, H, W] float32
    processed_dataset/<split>_dtms.{tar,zip}                  (when --archive is set)

I/O performance on Colab — READ THIS BEFORE TRAINING
----------------------------------------------------
The DTMs are tens of thousands of small ``.npy`` files (~35 GB total for the
full BraTS-PEDs train split at float32). Reading many tiny files *directly from
Google Drive* during training is catastrophic: Drive's per-file access latency
starves the A100 and the GPU sits idle. **Never point ``BraTSDataset`` at a
Drive path for the DTMs (or images/masks).**

Recommended workflow:

1. Run this script (once) writing the DTMs to Drive, with ``--archive tar``
   (or ``zip``) so each split's DTMs become a SINGLE archive on Drive.
2. On the Colab instance, copy the single archive from Drive to the local NVMe
   scratch disk (``/content/``) — one large sequential copy, not thousands of
   tiny reads::

       cp "/content/drive/MyDrive/.../train_dtms.tar" /content/
       mkdir -p /content/processed_dataset/train
       tar -xf /content/train_dtms.tar -C /content/processed_dataset/train

3. Point ``BraTSDataset(..., return_dtm=True, dtm_dir="/content/processed_dataset/train/dtms")``
   at the LOCAL path. Do the same for images/masks. All training-time reads then
   hit local NVMe, keeping the A100 fed.

Use ``tar`` (no compression) when the local disk has room and you want the
fastest extraction; use ``zip`` (light deflate) to shrink the Drive footprint
and the copy time at the cost of some CPU on extraction.

Notes
-----
* Class weights are computed on the TRAIN split only (the dataset statistic
  must not leak validation/test information). DTMs are derived purely from the
  ground-truth masks, so computing them for val/test introduces no leakage.
* The DTM is computed on the FULL ``ORIG_SIZE`` mask; the center crop is applied
  later at load time, consistently with images and masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import zipfile
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# Make the project root importable when run as ``python -m scripts.precompute_gsl_stats``
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.dataset import compute_signed_dtm
from src.losses import compute_global_class_weights

DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_dataset")
WEIGHTS_FILENAME = "gsl_class_weights.json"


# ---------------------------------------------------------------------------
# 1. Global class weights (Eq. 13)
# ---------------------------------------------------------------------------


def count_voxels_per_class(split_dir: str, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Sum voxel counts per class over all mask ``.npy`` files in a split.

    Args:
        split_dir:   Path to a split root (must contain a ``masks/`` subdir).
        num_classes: Number of classes ``C``.

    Returns:
        Int64 array ``[C]`` of total voxel counts per class.
    """
    msk_dir = os.path.join(split_dir, "masks")
    if not os.path.isdir(msk_dir):
        raise FileNotFoundError(f"masks/ not found in {split_dir!r}")

    files = sorted(f for f in os.listdir(msk_dir) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy masks in {msk_dir!r}")

    counts = np.zeros(num_classes, dtype=np.int64)
    for fname in tqdm(files, desc="  counting voxels", leave=False):
        msk = np.load(os.path.join(msk_dir, fname))
        binc = np.bincount(msk.reshape(-1), minlength=num_classes)
        counts[: len(binc)] += binc[:num_classes]
    return counts


def compute_and_save_weights(
    data_root: str,
    train_split: str = "train",
    num_classes: int = NUM_CLASSES,
) -> Dict:
    """Compute global w_k from the TRAIN split and save them as JSON.

    Returns the dict that was written (also useful for logging / tests).
    """
    split_dir = os.path.join(data_root, train_split)
    counts = count_voxels_per_class(split_dir, num_classes)
    weights = compute_global_class_weights(counts.tolist()).tolist()

    payload = {
        "source_split": train_split,
        "num_classes": num_classes,
        "class_names": list(CLASS_NAMES),
        "voxel_counts": counts.tolist(),
        "weights": weights,                       # w_k, Eq. 13 (normalised)
        "formula": "w_k = (1 / sum_j 1/N_j) * (1 / N_k)",
    }

    out_path = os.path.join(data_root, WEIGHTS_FILENAME)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Global class weights (Eq. 13), from '{train_split}' split:")
    for name, n, w in zip(CLASS_NAMES, counts.tolist(), weights):
        print(f"    {name:>10s}: N={n:>12,d}   w_k={w:.6f}")
    print(f"  Saved -> {out_path}")
    return payload


# ---------------------------------------------------------------------------
# 2. Per-slice signed DTMs (paper Fig. 2)
# ---------------------------------------------------------------------------


def precompute_dtms_for_split(
    split_dir: str,
    num_classes: int = NUM_CLASSES,
    overwrite: bool = False,
) -> int:
    """Compute and save the signed DTM for every mask ``.npy`` in a split.

    DTMs are written to ``<split_dir>/dtms/<same_filename>.npy`` with shape
    ``[C, H, W]`` float32.

    Args:
        split_dir:   Split root (must contain ``masks/``).
        num_classes: Number of classes.
        overwrite:   If False, skip files whose DTM already exists (idempotent
                     re-runs / resumable on Colab).

    Returns:
        Number of DTM files written.
    """
    msk_dir = os.path.join(split_dir, "masks")
    dtm_dir = os.path.join(split_dir, "dtms")
    if not os.path.isdir(msk_dir):
        raise FileNotFoundError(f"masks/ not found in {split_dir!r}")
    os.makedirs(dtm_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(msk_dir) if f.endswith(".npy"))
    n_written = 0
    for fname in tqdm(files, desc=f"  DTM [{os.path.basename(split_dir)}]", leave=False):
        out_path = os.path.join(dtm_dir, fname)
        if os.path.exists(out_path) and not overwrite:
            continue
        msk = np.load(os.path.join(msk_dir, fname))          # [H, W] int
        dtm = compute_signed_dtm(msk, num_classes=num_classes)  # [C, H, W] float32
        np.save(out_path, dtm)
        n_written += 1

    print(f"  {os.path.basename(split_dir):>6s}: {n_written} DTMs written "
          f"({len(files)} masks total) -> {dtm_dir}")
    return n_written


# ---------------------------------------------------------------------------
# 3. Archive DTMs into a single file (avoid Drive small-file I/O at train time)
# ---------------------------------------------------------------------------


def archive_split_dtms(
    split_dir: str,
    fmt: str,
    archive_dir: str | None = None,
) -> str:
    """Pack a split's ``dtms/`` directory into a single archive.

    Bundling the tens of thousands of tiny ``.npy`` DTMs into ONE file turns the
    training-time "thousands of small Drive reads" anti-pattern into a single
    large sequential copy: the archive is copied from Drive to the Colab local
    NVMe (``/content/``) once and extracted there (see the module docstring).

    Args:
        split_dir:   Split root containing ``dtms/`` (e.g.
                     ``processed_dataset/train``).
        fmt:         ``"tar"`` (no compression — fastest extraction) or
                     ``"zip"`` (light deflate — smaller Drive footprint).
        archive_dir: Directory to write the archive into. Defaults to the parent
                     of ``split_dir`` (i.e. next to the split folders).

    Returns:
        Path to the written archive.

    Raises:
        FileNotFoundError: If the ``dtms/`` directory does not exist.
        ValueError:        If ``fmt`` is not ``"tar"`` or ``"zip"``.
    """
    if fmt not in ("tar", "zip"):
        raise ValueError(f"Unsupported archive format {fmt!r}; use 'tar' or 'zip'.")

    dtm_dir = os.path.join(split_dir, "dtms")
    if not os.path.isdir(dtm_dir):
        raise FileNotFoundError(
            f"dtms/ not found in {split_dir!r}; run DTM pre-computation first."
        )

    split_name = os.path.basename(os.path.normpath(split_dir))
    out_dir = archive_dir if archive_dir is not None else os.path.dirname(
        os.path.normpath(split_dir)
    )
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(dtm_dir) if f.endswith(".npy"))
    # Archive members are stored under a top-level "dtms/" so extraction into a
    # split directory reproduces the exact layout BraTSDataset expects.
    arcroot = "dtms"

    if fmt == "tar":
        out_path = os.path.join(out_dir, f"{split_name}_dtms.tar")
        # Mode w writes an uncompressed tar (fastest to extract on the NVMe disk).
        with tarfile.open(out_path, "w") as tar:
            for fname in tqdm(files, desc=f"  archiving [{split_name}->tar]", leave=False):
                tar.add(os.path.join(dtm_dir, fname), arcname=os.path.join(arcroot, fname))
    else:  # zip
        out_path = os.path.join(out_dir, f"{split_name}_dtms.zip")
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for fname in tqdm(files, desc=f"  archiving [{split_name}->zip]", leave=False):
                zf.write(os.path.join(dtm_dir, fname), arcname=os.path.join(arcroot, fname))

    size_gb = os.path.getsize(out_path) / 1024 ** 3
    print(f"  {split_name:>6s}: archived {len(files)} DTMs -> {out_path}  ({size_gb:.2f} GB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Processed dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process for DTMs (default: train val test).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split used to compute the global class weights (default: train).",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only (re)compute the global class weights, skip DTM generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute DTMs even if the output file already exists.",
    )
    parser.add_argument(
        "--archive",
        choices=["none", "tar", "zip"],
        default="none",
        help="After computing DTMs, pack each split's dtms/ into a single "
             "archive to avoid Drive small-file I/O at train time: 'tar' "
             "(uncompressed, fastest extraction) or 'zip' (light deflate, "
             "smaller). Default: none.",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Directory to write archives into (default: next to the split "
             "folders, i.e. inside --data-root).",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  GSL offline pre-computation")
    print("=" * 70)

    # 1. Global class weights (always, from the train split)
    compute_and_save_weights(args.data_root, args.train_split)

    # 2. Per-slice DTMs (unless --weights-only)
    if not args.weights_only:
        print("\n  Pre-computing signed DTMs per slice ...")
        total = 0
        for split in args.splits:
            split_dir = os.path.join(args.data_root, split)
            if not os.path.isdir(split_dir):
                print(f"  [skip] split not found: {split_dir}")
                continue
            total += precompute_dtms_for_split(split_dir, overwrite=args.overwrite)
        print(f"\n  Total DTMs written: {total}")

    # 3. Optional archiving (runs even with --weights-only, so previously
    #    computed DTMs can be packed without recomputation).
    if args.archive != "none":
        print(f"\n  Packing DTMs into {args.archive} archive(s) "
              "(copy this to Colab local NVMe and extract there — see module docstring) ...")
        for split in args.splits:
            split_dir = os.path.join(args.data_root, split)
            dtm_dir = os.path.join(split_dir, "dtms")
            if not os.path.isdir(dtm_dir):
                print(f"  [skip] no dtms/ to archive in: {split_dir}")
                continue
            archive_split_dtms(split_dir, fmt=args.archive, archive_dir=args.archive_dir)

    print("\n[DONE] GSL pre-computation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
