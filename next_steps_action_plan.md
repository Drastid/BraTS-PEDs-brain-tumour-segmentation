# Next Steps Action Plan — BraTS-PEDs Brain Tumour Segmentation

**Author:** Code-Reviewer Skill
**Date:** 2026-05-11
**Companion document:** `current_progress_review.md`

This document lists every concrete remaining task to bring the project to **100% completion** relative to its original specification, plus optional improvements. Tasks are sorted by priority and grouped by category. Each task includes acceptance criteria so it is unambiguous when "done".

---

## Roadmap Overview

| # | Track | Estimated effort | Blockers |
|---|---|---|---|
| 1 | Implement missing IoU metric | 30 min | — |
| 2 | Move `import warnings` to module-level | 5 min | — |
| 3 | Centralise constants in `src/constants.py` | 1 hour | — |
| 4 | Add unit tests for losses, eval_utils, dataset | 2–3 hours | — |
| 5 | Re-run 3D test eval with IoU after Task 1 | 10 min | Task 1 |
| 6 | Save NIfTI predictions for clinical viewing | 1 hour | — |
| 7 | Reconcile output directories | 30 min | — |
| 8 | Delete unused venvs (`.venv-1` … `.venv-4`) | 5 min | User confirmation |
| 9 | Cross-domain zero-shot evaluation | half-day | Adult BraTS dataset |
| 10 | Final report polish (README expansion) | 1 hour | — |

Total focused effort to reach 100%: **~6–8 hours** of work, excluding Task 9 (data-dependent).

---

## TASK 1 — Implement `compute_iou_volume()` 🔴 PRIORITY A

**Why:** The project specification (`context.md` §1.5) explicitly lists IoU among the required clinical metrics: *"Dice per class, mean IoU, HD95"*. IoU is currently computed nowhere in the pipeline.

**Where:** `src/eval_utils.py`, added after `compute_dice_volume()`.

**Implementation:**
```python
def compute_iou_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-class Intersection-over-Union on 3D label volumes.

    Formula: IoU_c = (|P_c ∩ G_c| + ε) / (|P_c ∪ G_c| + ε)

    Returns Dict mapping ``"iou_<classname>"`` -> float in [0, 1].
    """
    results: Dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        pred_c = (pred == c)
        gt_c   = (gt   == c)
        inter  = float((pred_c & gt_c).sum())
        union  = float((pred_c | gt_c).sum())
        results[f"iou_{name}"] = (inter + smooth) / (union + smooth)
    return results
```

**Wire-up:** in `src/evaluate_3d_test.py`, inside `evaluate_3d_on_test()`:
1. Import `compute_iou_volume` at the top.
2. Call it after `compute_hd95_volume`:
   ```python
   iou = compute_iou_volume(pred_pp, gt_vol)
   record.update(iou)
   ```
3. Add IoU columns to the aggregate summary (mean/std/min/max per class).
4. Extend the JSON `summary` section with `iou_mean` and `iou_std` per class.

**Acceptance:** after running `python -m src.evaluate_3d_test --model all`, each `test_3d_metrics_*.json` file contains `iou_NCR`, `iou_ED`, `iou_ET` per subject AND per-class `iou_mean` / `iou_std` in the summary.

---

## TASK 2 — Hoist `import warnings` in `eval_utils.py` 🔴 PRIORITY A

**Why:** Cosmetic but visible. Currently imported three times inside the body of `compute_hd95_volume()` (cases B, C, and the bare-except). Should be at module level.

**Implementation:** in `src/eval_utils.py`:

1. Add `import warnings` at the top of the file (alphabetically next to `os`, `re`).
2. Remove the three `import warnings` statements at lines ~524, ~538, ~558 (look for them inside `compute_hd95_volume`).

**Acceptance:** `grep -n "import warnings" src/eval_utils.py` returns exactly one line — the module-level import.

---

## TASK 3 — Centralise constants in `src/constants.py` 🟡 PRIORITY B

**Why:** Right now `NUM_CLASSES = 4`, `CLASS_NAMES`, `CROP_SIZE = 192`, `ORIG_SIZE = 240`, `N_SLICES = 155`, and the BraTS voxel-frequency vector are duplicated in `dataset.py`, `train_utils.py`, `eval_utils.py`, and `evaluate_3d_test.py`. Any future cross-domain dataset with different spatial dims requires editing multiple files.

**Implementation:**

1. Create `src/constants.py`:
   ```python
   """Shared constants for the BraTS-PEDs pipeline."""
   from __future__ import annotations
   import numpy as np
   from typing import Tuple

   NUM_CLASSES: int = 4
   CLASS_NAMES: Tuple[str, ...] = ("background", "NCR", "ED", "ET")

   CROP_SIZE: int = 192
   ORIG_SIZE: int = 240
   N_SLICES: int = 155

   # BraTS-PEDs global voxel frequencies (from EDA on 257 subjects)
   VOXEL_FREQ = np.array([0.9940, 0.00066, 0.00441, 0.00091], dtype=np.float32)
   ```
2. In `src/dataset.py`: remove the local `NUM_CLASSES`, add `from .constants import NUM_CLASSES` (and `MODALITIES` stays local — it is dataset.py-specific).
3. In `src/train_utils.py`: remove the local `NUM_CLASSES`, `CLASS_NAMES`, `_VOXEL_FREQ`, add `from .constants import NUM_CLASSES, CLASS_NAMES, VOXEL_FREQ`. Rename `_VOXEL_FREQ` references to `VOXEL_FREQ`.
4. In `src/eval_utils.py`: remove the local `CROP_SIZE`, `ORIG_SIZE`, `N_SLICES`, `NUM_CLASSES`, `CLASS_NAMES`, add `from .constants import ...`. Keep the `_CROP_TOP` and `_CROP_LEFT` derived offsets (they are not constants per se).
5. In `src/evaluate_3d_test.py`: replace the local `NUM_CLASSES = 4` with the import.

**Acceptance:** `grep -rn "NUM_CLASSES = 4" src/` returns exactly one line — in `src/constants.py`. All training and evaluation scripts still run identically.

---

## TASK 4 — Add unit tests for critical functions 🟡 PRIORITY B

**Why:** No regression net. Any refactor (especially Task 3) is risky without tests. Five high-value tests:

**Implementation:** create `tests/` with the following files.

### `tests/test_losses.py`
- `test_dice_loss_perfect_prediction`: build `logits` that produce one-hot predictions matching `targets` → `DiceLoss` returns ~0.
- `test_dice_loss_inverted`: predictions of the wrong class → `DiceLoss` returns ~1.
- `test_focal_loss_zero_gamma_equals_ce`: with `γ=0` and no `alpha`, focal loss equals `F.cross_entropy(...)` on the same inputs (within float tolerance).

### `tests/test_eval_utils.py`
- `test_dice_perfect_volume`: identical synthetic volumes → all per-class Dice = 1.
- `test_remove_small_components`: planted 3 blobs (10, 60, 1000 voxels) → only the 60- and 1000-voxel ones survive `min_voxels=50`.
- `test_hd95_edge_cases`: both empty / GT-only / pred-only → all return `NaN`; normal case returns a positive float.
- `test_iou_perfect_volume` (after Task 1): identical synthetic volumes → all per-class IoU = 1.

### `tests/test_dataset.py`
- `test_clip_outliers`: a volume with a single voxel set to 1e6 and the rest in [0, 100] → after clipping at p99.5, the max becomes ~100.
- `test_zscore_non_zero_only`: a volume with half zeros, half values uniformly in [10, 20] → after Z-score, the zeros are still zero and the rest have μ≈0, σ≈1.

**Tooling:** use `pytest` (add to `requirements.txt`). Run with `pytest tests/`.

**Acceptance:** `pytest tests/ -v` passes 100% on a fresh clone.

---

## TASK 5 — Re-run 3D test evaluation with IoU 🟡 PRIORITY B (depends on Task 1)

**Why:** Once Task 1 is done, the saved JSONs in `evaluation_outputs/` are out of date (no IoU columns). Re-running produces canonical files.

**Implementation:**
```powershell
python -m src.evaluate_3d_test --model all
```

This rebuilds all three `test_3d_metrics_*.json` files with IoU included. The 3D predictions are deterministic given the checkpoints, so Dice and HD95 values should be byte-identical to the current files except for the new IoU keys.

**Acceptance:** the three JSON files in `evaluation_outputs/` now contain IoU keys; their commit timestamp updates; Dice and HD95 means are identical to the previous run (verify by diff or eye).

---

## TASK 6 — Save NIfTI predictions for clinical viewing 🟢 PRIORITY C

**Why:** `volume_to_nifti()` and `load_subject_nifti_meta()` already exist in `eval_utils.py`, but nothing uses them. For a clinical-impact narrative in the report, exporting `.nii.gz` predictions for a handful of subjects (best + worst per model) lets a radiologist load them into ITK-SNAP or 3D Slicer.

**Implementation:**

Add a new section to `08_comparison.ipynb` (or create `09_export_nifti.ipynb`):
1. Pick 6 test subjects: best-Dice and worst-Dice for each of the three models, deduplicated.
2. For each (model, subject), call `predict_volume()` then `remove_small_components()` then `volume_to_nifti()` with the affine from the raw NIfTI (`load_subject_nifti_meta`).
3. Save under `evaluation_outputs/nifti_predictions/{model}/{subject_id}_pred.nii.gz`.
4. Add a markdown cell listing the 6 subject IDs and Dice scores, with a note: "Load these in 3D Slicer or ITK-SNAP to inspect contour quality."

**Acceptance:** the `evaluation_outputs/nifti_predictions/` directory contains 6 `.nii.gz` files per model (best + worst Dice subjects) and they open correctly in a NIfTI viewer.

---

## TASK 7 — Reconcile output directories 🟢 PRIORITY C

**Why:** Right now training curves are split between `EDA_02_outputs/fpn_training_curves.png` and `processing_02_outputs/{unet,segformer}_training_curves.png`. There's no semantic reason for this — it's historical drift.

**Implementation:**

Option A (cleanest, but touches notebooks):
1. Create `training_outputs/` directory.
2. Move `EDA_02_outputs/fpn_training_curves.png` and `processing_02_outputs/*_training_curves.png` and `*_val_predictions.png` into it.
3. Update `06_train_fpn.ipynb`, `03_train_unet.ipynb`, `05_train_segformer.ipynb` to write to `training_outputs/`.
4. Delete the now-empty `EDA_02_outputs/`.

Option B (minimal): rename `EDA_02_outputs/` → `processing_02_outputs/` and consolidate. Update one path in `06_train_fpn.ipynb`.

**Acceptance:** all training curves and val-prediction visualisations live in one directory; no orphan directories remain.

---

## TASK 8 — Delete unused virtual environments 🟢 PRIORITY C

**Why:** `.venv-1`, `.venv-2`, `.venv-3`, `.venv-4` are dead artefacts from the Python 3.14 setup pain. Each occupies multiple GB on disk. They are in `.gitignore` so they do not affect git, but they consume local disk.

**Implementation (ask user before running):**
```powershell
Remove-Item -Recurse -Force .venv-1, .venv-2, .venv-3, .venv-4
```

**Acceptance:** only `.venv` remains; `python -c "import torch; print(torch.__version__)"` still works inside `.venv`.

**SAFETY:** confirm with the user before deletion. These are local-only, but irreversible.

---

## TASK 9 — Cross-domain zero-shot evaluation 🟢 PRIORITY C (data-dependent)

**Why:** `07_cross_domain_evaluation.ipynb` is fully written but never executed. Running it on an adult BraTS dataset would quantify the paediatric → adult domain shift in performance — a meaningful generalisation experiment that strengthens any final report.

**Blocker:** requires an adult BraTS dataset (e.g. BraTS2023-GLI, BraTS2024). Not available in the current working directory.

**Implementation (when data available):**
1. Place the adult NIfTI files under `PKG - BraTS-2023-GLI/` (or similar).
2. Re-run `02_preprocessing.ipynb` with `DATA_ROOT` and `OUTPUT_ROOT` pointed at the new dataset. Output goes to `processed_dataset_adult/`.
3. In `07_cross_domain_evaluation.ipynb`, set `NEW_DATASET_PATH = "processed_dataset_adult/test"` (or whichever split is appropriate).
4. Run all cells. The notebook is built to handle different spatial dims automatically (`infer_dataset_shape()`).
5. Compare in-domain (paediatric) vs cross-domain (adult) Dice/HD95 per model.

**Acceptance:** `07_cross_domain_evaluation.ipynb` is fully executed; a new section in `comparison_outputs/` or a new doc captures the in-domain vs cross-domain delta.

---

## TASK 10 — Final report polish 🟢 PRIORITY C

**Why:** README is good but could absorb the IoU results (after Task 5) and a one-sentence clarification on the 2D-vs-3D Dice denominator difference.

**Implementation:**
1. After Task 5: add an IoU column to the 3D Volumetric Metrics table in `README.md`.
2. Add a one-sentence note under the "2D Slice-Level Metrics" table explaining that 2D Dice averages over slices while 3D Dice reconstructs volumes — they are not directly comparable.
3. Add a "Skills" section to README mentioning the local Claude Code skill `/code-reviewer` (if it remains tracked). Optional.

**Acceptance:** README is updated; one git commit with message `docs: add IoU results and clarify 2D vs 3D metrics`.

---

## Bug Fixes To Apply (current state)

There are **no functional bugs** in the codebase. The three minor items below are correctness-adjacent or cosmetic only.

| # | Severity | File | Issue | Fix |
|---|---|---|---|---|
| B1 | Cosmetic | `src/eval_utils.py` | `import warnings` repeated 3× inside `compute_hd95_volume` | Task 2 |
| B2 | Documentation | `CLAUDE.md` §10 | Lists "IoU not implemented" as known issue | Closes after Task 1 |
| B3 | Hygiene | repo root | 5 venv directories (only `.venv` is active) | Task 8 (user-confirm) |

---

## Suggested Order of Execution

If executing in a single session, do tasks in this order:

1. **Task 2** (5 min — trivial cleanup, no downstream effect)
2. **Task 1** (30 min — implement IoU)
3. **Task 5** (10 min — re-run 3D eval to refresh JSONs with IoU)
4. **Task 10** (1 hr — update README with new IoU results)
5. **Task 3** (1 hr — centralise constants; test by re-running `evaluate_3d_test --model all` and confirming bit-identical Dice/HD95)
6. **Task 4** (2–3 hr — add unit tests; provides safety net for any future change)
7. **Task 6** (1 hr — NIfTI exports for clinical viewing)
8. **Task 7** (30 min — directory cleanup)
9. **Task 8** (5 min — venv cleanup, ask user first)
10. **Task 9** (if/when adult BraTS data available)

Each task ends with a self-contained commit (descriptive message) and a push.

---

## Summary

The project is functionally complete and produces strong, well-validated results (SegFormer-B1, 0.6760 mean-FG Dice on 3D test set). The remaining work is **opportunistic polish** rather than necessary completion:

- **Critical** (must do for spec compliance): Task 1 (IoU).
- **Strongly recommended**: Tasks 2, 3, 4, 5, 10.
- **Nice-to-have**: Tasks 6, 7, 8.
- **Stretch**: Task 9 (cross-domain, data-dependent).

After completing Tasks 1–5 and 10, the project moves from ~95% → 100% relative to its original specification. Tasks 6–9 add polish but are not part of the original deliverable.
