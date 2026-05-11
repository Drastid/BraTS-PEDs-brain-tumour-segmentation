"""
src/export_nifti_predictions.py
===============================
Export 3D segmentation predictions as NIfTI for clinical inspection.

For each of the three models (U-Net, FPN, SegFormer-B1), picks the best- and
worst-Dice test subjects (mean foreground Dice from the JSONs written by
``src.evaluate_3d_test``), re-runs prediction with post-processing, wraps the
result in a NIfTI image carrying the original affine and header, and saves it
under ``evaluation_outputs/nifti_predictions/{model}/``.

The output files are suitable for loading in 3D Slicer or ITK-SNAP alongside
the raw modalities under ``PKG - BraTS-PEDs-v1/...`` to visually inspect
contour quality at the tumour boundary.

Usage
-----
    python -m src.export_nifti_predictions
    python -m src.export_nifti_predictions --nifti-root "alt/path/to/Training"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch

# Make the project root importable when running ``python -m src.export_nifti_predictions``
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval_utils import (
    load_subject_nifti_meta,
    predict_volume,
    remove_small_components,
    volume_to_nifti,
)
from src.evaluate_3d_test import (
    DEVICE,
    MIN_COMPONENT_VOXELS,
    OUTPUT_DIR,
    TEST_DATA_DIR,
    load_fpn_model,
    load_segformer_model,
    load_unet_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NIFTI_ROOT = os.path.join(
    PROJECT_ROOT, "PKG - BraTS-PEDs-v1", "BraTS-PEDs-v1", "Training"
)
TUMOUR_CLASSES: Tuple[str, ...] = ("NCR", "ED", "ET")

MODEL_LOADERS = {
    "unet": load_unet_model,
    "fpn": load_fpn_model,
    "segformer": load_segformer_model,
}


# ---------------------------------------------------------------------------
# Subject selection
# ---------------------------------------------------------------------------


def _mean_fg_dice(record: Dict) -> float:
    """Mean of dice_NCR, dice_ED, dice_ET for one per_subject record."""
    return float(np.mean([record[f"dice_{c}"] for c in TUMOUR_CLASSES]))


def pick_best_worst(metrics_json: str) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    """Return (best_id, best_dice), (worst_id, worst_dice) from a metrics JSON."""
    with open(metrics_json, "r") as f:
        data = json.load(f)
    scored = [(r["subject_id"], _mean_fg_dice(r)) for r in data["per_subject"]]
    scored.sort(key=lambda x: x[1])
    return scored[-1], scored[0]


# ---------------------------------------------------------------------------
# Per-model export
# ---------------------------------------------------------------------------


def export_subject(
    model: torch.nn.Module,
    subject_id: str,
    model_name: str,
    nifti_root: str,
    out_root: Path,
) -> Path:
    """Run inference for one subject and save the post-processed prediction as NIfTI."""
    pred_vol, _ = predict_volume(model, TEST_DATA_DIR, subject_id, DEVICE)
    pred_pp = remove_small_components(pred_vol, min_voxels=MIN_COMPONENT_VOXELS)

    meta = load_subject_nifti_meta(nifti_root, subject_id)
    nii = volume_to_nifti(pred_pp, meta.affine, meta.header)

    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_pred.nii.gz"
    nib.save(nii, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(nifti_root: str) -> None:
    if not os.path.isdir(nifti_root):
        raise SystemExit(f"NIfTI source not found: {nifti_root!r}")

    out_root = Path(OUTPUT_DIR) / "nifti_predictions"

    # ── 1. Pick best / worst subjects per model ──────────────────────────
    picks: Dict[str, Tuple[Tuple[str, float], Tuple[str, float]]] = {}
    print("\n=== Picked subjects (per-model best & worst 3D mean fg Dice) ===")
    for model_name in MODEL_LOADERS:
        json_path = os.path.join(OUTPUT_DIR, f"test_3d_metrics_{model_name}.json")
        if not os.path.isfile(json_path):
            raise SystemExit(f"Missing metrics JSON: {json_path}")
        best, worst = pick_best_worst(json_path)
        picks[model_name] = (best, worst)
        print(
            f"  {model_name:>10}:  best = {best[0]} ({best[1]:.4f})"
            f"  |  worst = {worst[0]} ({worst[1]:.4f})"
        )

    # ── 2. Run inference and save NIfTI per (model, subject) ─────────────
    saved: List[Path] = []
    for model_name, ((best_id, _), (worst_id, _)) in picks.items():
        print(f"\n=== {model_name} -- loading checkpoint ===")
        model = MODEL_LOADERS[model_name]()
        for label, subj in [("best", best_id), ("worst", worst_id)]:
            print(f"  [{label}] {subj}: predicting...", flush=True)
            path = export_subject(model, subj, model_name, nifti_root, out_root)
            saved.append(path)
            print(f"  [{label}] saved ->{path.relative_to(Path(PROJECT_ROOT))}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[DONE] Exported {len(saved)} NIfTI predictions.")
    for p in saved:
        print(f"  {p.relative_to(Path(PROJECT_ROOT))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nifti-root",
        default=DEFAULT_NIFTI_ROOT,
        help="Root directory containing per-subject NIfTI folders "
             "(default: %(default)s).",
    )
    args = parser.parse_args()
    main(args.nifti_root)
