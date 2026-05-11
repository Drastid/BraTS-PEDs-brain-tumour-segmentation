# Current Progress Review — BraTS-PEDs Brain Tumour Segmentation

**Reviewer:** Code-Reviewer Skill (refresh)
**Date:** 2026-05-11
**Scope:** Full codebase audit after TASK 1–7 + TASK 10 completion
**Previous review:** same date, earlier in the day (now superseded — listed everything as 🔴/🟡/🟢 to do)

---

## 1. Executive Summary

The project is now in a **fully shippable state — ~99% complete**. Every task from the prior action plan (1, 2, 3, 4, 5, 6, 7, 10) was executed and pushed to `origin/master` in 8 well-scoped commits. The only remaining item from the roadmap is **TASK 9 (cross-domain zero-shot evaluation)**, which is blocked on the availability of an adult BraTS dataset.

**Headline change since the last review:**
- ✅ The single missing spec metric (IoU) is now implemented, wired through the pipeline, and surfaced in `README.md`.
- ✅ The cosmetic `import warnings` defect is fixed.
- ✅ Constants are centralised in `src/constants.py`.
- ✅ A `tests/` directory with 9 pytest unit tests (passing in 4 s) provides a regression net.
- ✅ Six NIfTI predictions are exported for clinical viewing (`evaluation_outputs/nifti_predictions/`).
- ✅ Output directories reconciled — training artefacts now live in a single `training_outputs/` directory.
- ✅ README has IoU rows + a denominator-difference clarification on 2D vs 3D Dice.

**Winner (unchanged):** SegFormer-B1 — mean foreground 3D Dice **0.6760**, mean foreground 3D IoU **0.6047**.

---

## 2. Component-by-Component Review

### 2.1 `src/dataset.py`
**Verdict:** Excellent. Unchanged on substance.

- Local `NUM_CLASSES` removed; now imported from `src.constants` (re-exported for callers via `# noqa: F401`).
- `clip_outliers`, `zscore_normalise`, `load_subject` covered by `tests/test_dataset.py` (2 tests).

### 2.2 `src/losses.py`
**Verdict:** Excellent. Unchanged.

- DiceLoss / FocalLoss / CombinedLoss now have unit-test coverage (`tests/test_losses.py`: perfect-prediction, inverted-prediction, γ=0 equals cross-entropy).

### 2.3 `src/train_utils.py`
**Verdict:** Strong. Refactored cleanly.

- `NUM_CLASSES`, `CLASS_NAMES`, `_VOXEL_FREQ` removed; now imported from `src.constants` (renamed `_VOXEL_FREQ` → `VOXEL_FREQ`).
- Verified: class weights produced by `get_class_weights()` are byte-identical to the documented values (BG≈0.0004, NCR≈0.533, ED≈0.080, ET≈0.387).

### 2.4 `src/eval_utils.py`
**Verdict:** Most-improved module of this review.

- **NEW:** `compute_iou_volume()` — per-class IoU with the same shape and smoothing convention as `compute_dice_volume()`.
- **FIXED:** the 3× `import warnings` repetition inside `compute_hd95_volume()` is gone; one module-level import.
- Constants (`CROP_SIZE`, `ORIG_SIZE`, `N_SLICES`, `NUM_CLASSES`, `CLASS_NAMES`) imported from `src.constants`; derived offsets `_CROP_TOP`/`_CROP_LEFT` kept local.
- 4 unit tests in `tests/test_eval_utils.py` cover Dice/IoU perfect-volume, `remove_small_components`, HD95 edge cases.

### 2.5 `src/models.py`
**Verdict:** Unchanged. The 4-channel patch-embedding adaptation remains elegant.

### 2.6 `src/evaluate_3d_test.py`
**Verdict:** Solid, now writes IoU.

- IoU computed per subject and aggregated into the summary (`iou_mean`, `iou_std`, `mean_fg_iou`).
- Constants imported from `src.constants`; local `NUM_CLASSES = 4` removed.
- Test JSONs (`test_3d_metrics_{unet,fpn,segformer}.json`) regenerated on CUDA with the new IoU keys at commit `1dd54e3`.

### 2.7 **NEW** — `src/constants.py`
**Verdict:** Clean. Single source of truth for `NUM_CLASSES`, `CLASS_NAMES`, `CROP_SIZE`, `ORIG_SIZE`, `N_SLICES`, `VOXEL_FREQ`. Eliminates the duplication risk flagged in the previous review.

### 2.8 **NEW** — `src/export_nifti_predictions.py`
**Verdict:** Self-contained CLI that picks the best/worst test subject per model from the metrics JSONs, re-runs `predict_volume + remove_small_components`, wraps the output with the original affine via `load_subject_nifti_meta + volume_to_nifti`, and saves 6 `.nii.gz` files under `evaluation_outputs/nifti_predictions/{model}/`. Re-uses everything from `eval_utils` and `evaluate_3d_test`; no duplication.

### 2.9 **NEW** — `tests/`
**Verdict:** A modest but real regression net.

- 9 tests across 3 files: `test_losses.py` (3), `test_eval_utils.py` (4), `test_dataset.py` (2). All pass in ≈ 4 seconds on CPU. `pytest.ini` sets `pythonpath = .` and `testpaths = tests` for one-command CI-style runs.

### 2.10 Notebooks
**Verdict:** Sync'd to the new output convention.

- All 5 notebooks that previously wrote to `EDA_02_outputs/` (`03_train_unet`, `04_evaluation`, `05_train_segformer`, `06_train_fpn`, `07_cross_domain_evaluation`) now point at `training_outputs/` in their source cells.
- Stale cell outputs intentionally left unchanged — they're historical and will refresh naturally on the next run.

### 2.11 Output directories — reconciled
**Verdict:** Clean. No more historical drift.

| Directory | Contents |
|---|---|
| `EDA_01_outputs/` | 4 EDA plots from `01_EDA.ipynb` |
| `processing_02_outputs/` | 5 `preproc_*.png` only (preprocessing QC) |
| `training_outputs/` | 8 PNGs (training curves, val predictions, failure analysis) |
| `evaluation_outputs/` | 3 JSON test-metric files + `nifti_predictions/` (6 `.nii.gz` + README) |
| `comparison_outputs/` | 5 PNG + 2 CSV from `08_comparison.ipynb` |
| ~~`EDA_02_outputs/`~~ | **deleted** |

### 2.12 Documentation
**Verdict:** Up to date.

- `README.md` now reports IoU alongside Dice and HD95 in the 3D Volumetric Metrics table; headline narrative cites the +11.5% IoU improvement of SegFormer over U-Net; 2D-vs-3D note explicitly states the denominator difference.
- `requirements.txt` gained `pytest`.
- `CLAUDE.md` §8 is stale — points to TASKs 1–3 as "highest-priority", all of which are now closed (see action plan).

---

## 3. Results — quantitative summary

3D test set (26 subjects, post-processed; from `evaluation_outputs/test_3d_metrics_*.json`):

| Metric | U-Net | FPN | SegFormer |
|---|---|---|---|
| Dice NCR | 0.5856 | 0.5590 | **0.6752** |
| Dice ED | 0.7239 | 0.7104 | **0.7229** |
| Dice ET | 0.5398 | 0.5946 | **0.6299** |
| **Mean FG Dice** | 0.6164 | 0.6213 | **0.6760** |
| IoU NCR | 0.5103 | 0.4940 | **0.5941** |
| IoU ED | 0.6162 | 0.5947 | **0.6173** |
| IoU ET | 0.4999 | 0.5657 | **0.6027** |
| **Mean FG IoU** | 0.5421 | 0.5515 | **0.6047** |
| HD95 NCR (mm) | 9.21 | 8.70 | **5.23** |
| HD95 ED (mm) | **7.45** | 7.63 | 8.15 |
| HD95 ET (mm) | 17.85 | 18.69 | **14.60** |

---

## 4. Recommendations & Refactoring

Everything from the prior 🔴/🟡 list is now done. The remaining items are either external blockers or genuinely optional polish.

### 4.1 🔴 Blocker (data-dependent)
- **TASK 9 — Cross-domain zero-shot evaluation.** `07_cross_domain_evaluation.ipynb` is structured and unblocked code-wise (notebook source now points at `training_outputs/` and handles arbitrary spatial dims via `infer_dataset_shape`). Execution remains blocked until an adult BraTS dataset (e.g. BraTS2023-GLI) lands under `PKG - BraTS-2023-GLI/`.

### 4.2 🟢 Optional polish
- **Update `CLAUDE.md` §8.** Current text still claims IoU / `import warnings` / constants centralisation are the "highest priority remaining". They are not — they're done. (Will be fixed in the same commit as this review.)
- **Update `project_review.md`** date header — it's the 2026-04-29 audit; everything 🔴 in it is already resolved. Leave the file as a historical record, but add a one-line "Status: superseded by 2026-05-11 reviews" at the top.
- **Consider deleting `.venv-1` and `.venv-2`.** They have CUDA torch but no `transformers`/`accelerate`, so they can't run SegFormer. They're effectively dead. **Do not delete `.venv-3` or `.venv-4`** — those are the active CUDA environments used for the GPU runs in TASK 5 / 6 (per memory `project_venvs.md`).
- **`07_cross_domain_evaluation.ipynb` stale outputs cleanup.** The notebook source now writes to `training_outputs/`, but its old cell outputs still mention `EDA_02_outputs/`. They'll auto-clear on the next execution; no action needed unless someone reads the stale outputs and gets confused.

### 4.3 ❌ Items not worth doing
- Renaming `processing_02_outputs/` → `preprocessing_outputs/` — adds churn across 5 preprocessing-related sources, low benefit.
- Adding integration tests that actually load checkpoints — slow, requires `.pth` files which are git-ignored, and the existing 9 unit tests plus the end-to-end `evaluate_3d_test --model all` already provide good coverage.
- Refactoring `train_utils.evaluate()` to share more code with `evaluate_3d_test.evaluate_3d_on_test()` — they intentionally do different things (2D slice-level vs 3D volumetric); merging would obscure both.

---

## 5. Critical Invariants (must not be violated by any future change)

Unchanged from the previous review and from `CLAUDE.md` §2. Listed here for completeness:

1. **Augmentation = geometric only** (HFlip, Rotate90, ShiftScaleRotate, ElasticTransform). No brightness/contrast/jitter.
2. **Center crop 192×192 at `top=24, left=24`**, identical at training and inference.
3. **Label remap `seg == 4 → 3`** in `dataset.load_subject()`.
4. **Z-score on non-zero voxels only**; background re-zeroed after normalisation.
5. **HD95 NaN convention** — both-empty / GT-only / pred-only → `np.nan`; downstream uses `np.nanmean`.
6. **Split frozen** in `split.json` with `SEED=42` (205 train / 26 val / 26 test).

---

## 6. Overall Assessment

| Aspect | Rating | Change vs prior review |
|---|---|---|
| Code architecture | ★★★★★ | unchanged |
| Documentation | ★★★★★ | unchanged |
| Spec adherence | ★★★★★ | unchanged (was already 5★) |
| Pipeline completeness | ★★★★★ | ↑ from 4★ — IoU now implemented |
| Result quality | ★★★★☆ | unchanged |
| Reproducibility | ★★★★★ | unchanged |
| Code quality | ★★★★★ | ↑ from 4★ — `import warnings` fixed, constants centralised |
| Testing | ★★★★☆ | ↑ from 0★ — 9 pytest tests added |
| Clinical artefacts | ★★★★★ | ↑ from 3★ — 6 NIfTI predictions exported for ITK-SNAP / 3D Slicer |

**Verdict:** Project is in a **complete and shippable state (~99%)**. The only remaining roadmap item is data-dependent (TASK 9 cross-domain). All code-quality, testing, documentation, and clinical-output deliverables are in place.
