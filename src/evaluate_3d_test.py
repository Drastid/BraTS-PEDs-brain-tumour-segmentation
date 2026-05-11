"""
src/evaluate_3d_test.py
=======================
3D volumetric evaluation script for the BraTS-PEDs segmentation pipeline.

Runs the full 3D clinical evaluation (Dice + HD95 per class, with
post-processing) on the **TEST set** for U-Net, FPN, and SegFormer models.

This script addresses action item 5.1 from project_review.md:
    "Implementare la valutazione 3D sul TEST set"

Usage
-----
    python -m src.evaluate_3d_test --model unet
    python -m src.evaluate_3d_test --model fpn
    python -m src.evaluate_3d_test --model segformer
    python -m src.evaluate_3d_test --model all

Results are printed to stdout and saved as JSON files in
``evaluation_outputs/`` for later use in the comparison notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports work when run as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.eval_utils import (
    get_subject_ids,
    predict_volume,
    remove_small_components,
    compute_dice_volume,
    compute_hd95_volume,
    compute_iou_volume,
)
from src.train_utils import load_checkpoint
# NOTE: src.models.get_segformer is imported lazily inside load_segformer_model()
# to avoid pulling in the 'transformers' dependency when only evaluating U-Net.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processed dataset — TEST split
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_dataset", "test")

# Checkpoints
UNET_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "unet", "best.pth")
FPN_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "fpn", "best.pth")
SEGFORMER_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "segformer", "best.pth")

# Model architecture constants (must match training configuration)
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 4

# Post-processing
MIN_COMPONENT_VOXELS = 50

# BraTS-PEDs isotropic voxel spacing (mm)
VOXEL_SPACING = (1.0, 1.0, 1.0)

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_outputs")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_unet_model() -> torch.nn.Module:
    """Load the best U-Net checkpoint and return the model in eval mode."""
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,   # weights loaded from checkpoint
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(DEVICE)

    ckpt = load_checkpoint(UNET_CHECKPOINT, model, device=DEVICE)
    epoch = ckpt["epoch"]
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  U-Net checkpoint loaded: epoch {epoch}, {n_params:,} params")
    return model


def load_fpn_model() -> torch.nn.Module:
    """Load the best FPN checkpoint and return the model in eval mode."""
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=None,   # weights loaded from checkpoint
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(DEVICE)

    ckpt = load_checkpoint(FPN_CHECKPOINT, model, device=DEVICE)
    epoch = ckpt["epoch"]
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FPN checkpoint loaded: epoch {epoch}, {n_params:,} params")
    return model


def load_segformer_model() -> torch.nn.Module:
    """Load the best SegFormer checkpoint and return the model in eval mode."""
    from src.models import get_segformer  # lazy import to avoid 'transformers' dependency

    model = get_segformer(
        model_checkpoint="nvidia/mit-b1",
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    ckpt = load_checkpoint(SEGFORMER_CHECKPOINT, model, device=DEVICE)
    epoch = ckpt["epoch"]
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SegFormer checkpoint loaded: epoch {epoch}, {n_params:,} params")
    return model


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_3d_on_test(
    model: torch.nn.Module,
    model_name: str,
) -> pd.DataFrame:
    """Run full 3D volumetric evaluation on the test set.

    For each test subject:
    1. Run predict_volume (slice-by-slice, with centre-crop consistency)
    2. Apply remove_small_components (26-conn, threshold 50 voxels)
    3. Compute 3D Dice and HD95 per class
    4. Accumulate results

    Args:
        model:      Segmentation model in eval mode, on DEVICE.
        model_name: Human-readable name for logging ("U-Net" or "SegFormer").

    Returns:
        DataFrame with per-subject metrics (subject_id as index).
    """
    subject_ids = get_subject_ids(TEST_DATA_DIR)
    print(f"\n{'=' * 70}")
    print(f"  {model_name} — 3D EVALUATION ON TEST SET  (n={len(subject_ids)} subjects)")
    print(f"{'=' * 70}")

    records: List[Dict] = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)

        for i, subject_id in enumerate(subject_ids):
            # ── 1. 3D inference ──
            pred_raw, gt_vol = predict_volume(
                model, TEST_DATA_DIR, subject_id, DEVICE
            )

            # ── 2. Post-processing ──
            pred_pp = remove_small_components(pred_raw, min_voxels=MIN_COMPONENT_VOXELS)

            # ── 3. Dice (on post-processed prediction) ──
            dice = compute_dice_volume(pred_pp, gt_vol)

            # ── 4. HD95 (on post-processed prediction) ──
            hd95 = compute_hd95_volume(pred_pp, gt_vol, voxel_spacing=VOXEL_SPACING)

            # ── 5. IoU (on post-processed prediction) ──
            iou = compute_iou_volume(pred_pp, gt_vol)

            # ── 6. Accumulate ──
            record = {"subject_id": subject_id}
            record.update(dice)
            record.update(hd95)
            record.update(iou)
            records.append(record)

            # Progress
            print(f"  [{i+1}/{len(subject_ids)}] {subject_id} done", end="\r")

    # Print edge-case warnings summary
    print()  # newline after progress
    if caught_warnings:
        print(f"\n  [Edge-case warnings: {len(caught_warnings)}]")
        for w in caught_warnings:
            print(f"    {w.category.__name__}: {w.message}")
    else:
        print("\n  No edge-case warnings — all HD95 values computed normally.")

    # Build DataFrame
    df = pd.DataFrame(records).set_index("subject_id")

    # ── Aggregate statistics ──
    TUMOUR_CLASSES = ["NCR", "ED", "ET"]
    summary_rows = []
    for cls in TUMOUR_CLASSES:
        dice_col = f"dice_{cls}"
        hd95_col = f"hd95_{cls}"
        iou_col  = f"iou_{cls}"
        dice_vals = df[dice_col].values
        hd95_vals = df[hd95_col].values
        iou_vals  = df[iou_col].values
        n_hd95_nan = int(np.sum(np.isnan(hd95_vals)))

        summary_rows.append({
            "Class":          cls,
            "Dice mean":      f"{np.nanmean(dice_vals):.4f}",
            "Dice std":       f"{np.nanstd(dice_vals):.4f}",
            "Dice min":       f"{np.nanmin(dice_vals):.4f}",
            "Dice max":       f"{np.nanmax(dice_vals):.4f}",
            "IoU mean":       f"{np.nanmean(iou_vals):.4f}",
            "IoU std":        f"{np.nanstd(iou_vals):.4f}",
            "HD95 mean (mm)": f"{np.nanmean(hd95_vals):.2f}" if n_hd95_nan < len(hd95_vals) else "N/A",
            "HD95 std (mm)":  f"{np.nanstd(hd95_vals):.2f}"  if n_hd95_nan < len(hd95_vals) else "N/A",
            "HD95 NaN count": n_hd95_nan,
        })

    fg_dice_cols = [f"dice_{c}" for c in TUMOUR_CLASSES]
    fg_iou_cols  = [f"iou_{c}"  for c in TUMOUR_CLASSES]
    mean_fg_dice = np.nanmean(df[fg_dice_cols].values)
    mean_fg_iou  = np.nanmean(df[fg_iou_cols].values)

    summary_df = pd.DataFrame(summary_rows).set_index("Class")

    print(f"\n{'=' * 70}")
    print(f"  {model_name} — TEST SET — 3D CLINICAL METRICS  (n={len(df)} subjects)")
    print(f"{'=' * 70}")
    print(summary_df.to_string())
    print()
    print(f"  Mean foreground Dice (3D, post-processed): {mean_fg_dice:.4f}")
    print(f"  Mean foreground IoU  (3D, post-processed): {mean_fg_iou:.4f}")
    print(f"{'=' * 70}")

    return df


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(df: pd.DataFrame, model_name: str) -> str:
    """Save per-subject results as a JSON file.

    Args:
        df:         DataFrame with per-subject metrics.
        model_name: "unet" or "segformer".

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"test_3d_metrics_{model_name}.json")

    # Convert DataFrame to a serialisable format
    result = {
        "model": model_name,
        "split": "test",
        "n_subjects": len(df),
        "per_subject": df.reset_index().to_dict(orient="records"),
        "summary": {},
    }

    # Add summary statistics
    TUMOUR_CLASSES = ["NCR", "ED", "ET"]
    for cls in TUMOUR_CLASSES:
        dice_vals = df[f"dice_{cls}"].values
        hd95_vals = df[f"hd95_{cls}"].values
        iou_vals  = df[f"iou_{cls}"].values
        result["summary"][cls] = {
            "dice_mean": float(np.nanmean(dice_vals)),
            "dice_std":  float(np.nanstd(dice_vals)),
            "iou_mean":  float(np.nanmean(iou_vals)),
            "iou_std":   float(np.nanstd(iou_vals)),
            "hd95_mean": float(np.nanmean(hd95_vals)) if not np.all(np.isnan(hd95_vals)) else None,
            "hd95_std":  float(np.nanstd(hd95_vals))  if not np.all(np.isnan(hd95_vals)) else None,
            "hd95_nan_count": int(np.sum(np.isnan(hd95_vals))),
        }

    fg_dice_cols = [f"dice_{c}" for c in TUMOUR_CLASSES]
    fg_iou_cols  = [f"iou_{c}"  for c in TUMOUR_CLASSES]
    result["summary"]["mean_fg_dice"] = float(np.nanmean(df[fg_dice_cols].values))
    result["summary"]["mean_fg_iou"]  = float(np.nanmean(df[fg_iou_cols].values))

    # Handle NaN values for JSON serialization
    def nan_to_none(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [nan_to_none(v) for v in obj]
        return obj

    result = nan_to_none(result)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="3D volumetric evaluation on the BraTS-PEDs TEST set."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "fpn", "segformer", "all"],
        default="all",
        help="Which model to evaluate: 'unet', 'fpn', 'segformer', or 'all' (default).",
    )
    args = parser.parse_args()

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"SMP      : {smp.__version__}")
    print(f"Device   : {DEVICE}")
    print(f"Test dir : {TEST_DATA_DIR}")
    print()

    models_to_evaluate = []

    if args.model in ("unet", "all"):
        models_to_evaluate.append(("unet", "U-Net/ResNet34", load_unet_model))

    if args.model in ("fpn", "all"):
        models_to_evaluate.append(("fpn", "FPN/ResNet34", load_fpn_model))

    if args.model in ("segformer", "all"):
        models_to_evaluate.append(("segformer", "SegFormer-B1", load_segformer_model))

    for model_key, model_display_name, loader_fn in models_to_evaluate:
        print(f"\n{'#' * 70}")
        print(f"  Loading {model_display_name}...")
        print(f"{'#' * 70}")

        model = loader_fn()
        df = evaluate_3d_on_test(model, model_display_name)
        save_results(df, model_key)

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    print("\n[DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
