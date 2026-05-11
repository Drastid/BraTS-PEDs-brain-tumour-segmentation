# Current Progress Review — BraTS-PEDs Brain Tumour Segmentation

**Reviewer:** Code-Reviewer Skill
**Date:** 2026-05-11
**Scope:** Full codebase audit (`src/`, notebooks, outputs, docs, GitHub repo state)
**Reference inputs:** `context.md` (logbook), `project_review.md` (audit 2026-04-29), `CLAUDE.md` (project rules), `README.md`

---

## 1. Executive Summary

The project is in a **complete and shippable state**. All five planned development phases are done:

1. EDA on the BraTS-PEDs-v1 dataset (`01_EDA.ipynb`).
2. 3D→2D extraction, normalisation, stratified split (`02_preprocessing.ipynb`, `processed_dataset/`).
3. Three full training pipelines (`03_train_unet`, `05_train_segformer`, `06_train_fpn`) producing `best.pth` + `last.pth` + `history.json` for each model.
4. 3D volumetric clinical evaluation on the held-out test set (`src/evaluate_3d_test.py`, `evaluation_outputs/test_3d_metrics_{unet,fpn,segformer}.json`).
5. Three-way comparison (`08_comparison.ipynb`, `comparison_outputs/`).

**Winner:** SegFormer-B1 with mean foreground Dice (3D, post-processed) of **0.6760** — +9.7% over U-Net (0.6164), with 44% fewer parameters.

The codebase is well-structured, the `src/` modules have professional-grade docstrings and edge-case handling, the training is reproducible (SEED=42 saved in `split.json`), and the GitHub repository is initialised and up to date. Outstanding items are limited to (a) cross-domain execution (blocked by external data), (b) IoU metric (mentioned in spec but never implemented), and (c) minor code-quality cleanups.

---

## 2. Component-by-Component Review

### 2.1 `src/dataset.py` — Data I/O and preprocessing
**Verdict:** Excellent. Production-quality.

- `clip_outliers()` and `zscore_normalise()` are correctly scoped to non-zero (brain) voxels, with the background voxel re-zeroed after normalisation. This is the medically correct behaviour for MRI and is documented in the docstring.
- `load_subject()` performs the label remap `seg == 4 → 3` (old BraTS ET convention), so any future BraTS dataset using the old convention is handled.
- `BraTSDataset` uses mmap-backed `np.load` on pre-extracted `.npy` slices and gives the measured 38.6 ms/batch — 130–390× faster than live NIfTI parsing.
- The albumentations integration uses `additional_targets={"mask": "mask"}` and transposes `[4, H, W] → [H, W, 4]` so it works with multichannel float32 input.

**Minor note:** `NUM_CLASSES = 4` is declared here and re-declared in `train_utils.py` and `eval_utils.py`. Same for `CLASS_NAMES`. (See §3.3 below.)

### 2.2 `src/losses.py` — DiceLoss + FocalLoss + CombinedLoss
**Verdict:** Excellent.

- `DiceLoss` correctly excludes the background class when `ignore_background=True`. Laplace smoothing (`ε=1e-6`) keeps the gradient finite for empty classes.
- `FocalLoss` is numerically stable: `log_softmax` + `gather` for the per-pixel probability of the true class, rather than the naive `log(softmax(...))`. The `alpha` (per-class weight) and `gamma` parameters are properly registered as buffers.
- `CombinedLoss.forward()` returns a 3-tuple `(total, d_loss, f_loss)` — this enables the training loop to log the two raw components independently, which is genuinely useful when diagnosing convergence (e.g. the "click" at epoch 15 of U-Net training).

**Subtle gotcha:** when `ignore_background=True`, the per-class weights are intentionally not applied to the Dice term (`class_weights if not ignore_background else None`). That is correct (excluding BG already focuses on tumour classes), but worth keeping in mind if `ignore_background` ever toggles off.

### 2.3 `src/train_utils.py` — Training loop, augmentation, checkpoints
**Verdict:** Strong. The augmentation policy is exemplary.

- `get_augmentation(p=0.5)` is **strictly geometric** (HFlip, Rotate90, ShiftScaleRotate with `border_mode=0`, ElasticTransform at half probability). This is the right choice given Z-score-normalised float32 inputs with negative values — pixel-level brightness/contrast augmentation would corrupt the normalisation. The docstring warns explicitly.
- `_compute_batch_dice()` runs under `torch.no_grad()` so it adds no autograd overhead.
- `center_crop()` is deterministic: `top=24, left=24` from 240×240 → 192×192. Same offsets are used in `eval_utils.predict_volume()`, which is critical (see §4.1).
- `set_encoder_trainable()` uses `getattr(model, "encoder", None)` and raises a clear error if missing — works for `smp.Unet`, `smp.FPN`, and `SegFormerWrapper` (which exposes `.encoder`).
- AMP is enabled when `scaler` is provided; gradient clipping at `max_norm=1.0` is always applied.
- `save_checkpoint()` / `load_checkpoint()` serialise model + optimizer + scheduler + AMP scaler + epoch + metrics.

**Minor issue:** the constants `NUM_CLASSES = 4` and `CLASS_NAMES = ("background", "NCR", "ED", "ET")` are duplicated from `dataset.py`/`eval_utils.py`. Same `_VOXEL_FREQ` is hardcoded here while EDA produced these numbers — fine, but it means changes to dataset stats would need editing in multiple places.

### 2.4 `src/eval_utils.py` — 3D reconstruction, post-processing, clinical metrics
**Verdict:** Most sophisticated module in the project. Exemplary.

- `predict_volume()` mirrors the training-time center crop **exactly** and un-crops back into a 240×240 canvas before stacking slices. Cross-domain users get auto-detected `orig_size` / `n_slices` via `infer_dataset_shape()`.
- `remove_small_components()` uses 26-connectivity (`scipy.ndimage.generate_binary_structure(3, 3)`), iterates over `NCR`/`ED`/`ET` separately, and removes blobs smaller than `min_voxels=50`. This kills the "confetti effect" typical of 2D slice-based models.
- `compute_dice_volume()` returns per-class Dice on the full 3D volume with Laplace smoothing.
- `compute_hd95_volume()` is the most carefully designed function in the codebase. It handles three distinct degenerate cases explicitly (both empty / GT-only / pred-only) and returns `NaN` with `RuntimeWarning` for the latter two, so the downstream code can use `np.nanmean` and count failures separately. The wrapper catches any other medpy exception, also returning `NaN`.

**Defect (cosmetic):** `import warnings` is repeated three times inside the body of `compute_hd95_volume()` (cases B, C, and the exception handler) instead of being moved to module-level imports at the top of the file. This is the single most visible code-quality issue in `src/`. It does not affect correctness.

### 2.5 `src/models.py` — SegFormer 4-channel adaptation
**Verdict:** Elegant.

- The patch-embedding expansion from 3→4 input channels keeps the original RGB weights byte-for-byte and initialises the 4th channel as `old_weight.mean(dim=1, keepdim=True)`. This is a well-motivated initialisation: the new channel starts with a sensible "average RGB" response rather than random noise, so fine-tuning has a warm start.
- `SegFormerWrapper` exposes `.encoder` and `.decode_head` properties so `set_encoder_trainable()` works without modification and the two-phase optimizer (differential LR) plugs in cleanly.
- `forward()` upsamples logits from `[B, C, H/4, W/4]` to `[B, C, H, W]` via bilinear interpolation so `CombinedLoss` and `_compute_batch_dice()` see the expected spatial resolution.

### 2.6 `src/evaluate_3d_test.py` — CLI evaluation script
**Verdict:** Solid. Reproducible, hermetic, supports `--model all`.

- Three loader functions (`load_unet_model`, `load_fpn_model`, `load_segformer_model`) each rebuild the architecture, load the matching checkpoint, set `eval()`, and report params.
- Lazy-imports `src.models.get_segformer` so users can evaluate U-Net or FPN without installing `transformers`.
- `evaluate_3d_on_test()` runs the full pipeline (predict → post-process → Dice → HD95) and captures `RuntimeWarning`s into a list it prints at the end (good UX for the "complete miss" / "hallucination" edge cases).
- `save_results()` recursively converts `NaN` → `None` so the JSON files are valid (otherwise `json.dump` writes `NaN` tokens that some parsers reject).
- `torch.cuda.empty_cache()` is called between models, important because the 8GB-VRAM 3080 is tight.

### 2.7 Notebooks (workflow)
**Verdict:** Complete. Output artefacts present on disk.

| Notebook | State | Output |
|---|---|---|
| `01_EDA.ipynb` | Executed | 4 plots in `EDA_01_outputs/` |
| `02_preprocessing.ipynb` | Executed | `processed_dataset/` (39,538 slices), `split.json`, QC plots |
| `03_train_unet.ipynb` | Executed | `checkpoints/unet/{best,last}.pth` + history |
| `04_evaluation.ipynb` | Executed | Val-set 3D eval for U-Net (superseded by `evaluate_3d_test.py`) |
| `05_train_segformer.ipynb` | Executed | `checkpoints/segformer/{best,last}.pth` + history |
| `06_train_fpn.ipynb` | Executed | `checkpoints/fpn/{best,last}.pth` + history |
| `06b_evaluation_fpn.ipynb` | Executed | Calls `src.evaluate_3d_test --model fpn`, displays summary |
| `07_cross_domain_evaluation.ipynb` | Template, not executed | Blocked on external adult BraTS dataset |
| `08_comparison.ipynb` | Executed | `comparison_outputs/` (5 PNG + 2 CSV) |

### 2.8 Documentation
**Verdict:** Strong but slightly redundant.

- `CLAUDE.md` (11 sections) — comprehensive single-source-of-truth for an LLM agent operating on the repo. The critical-constraints section (augmentation policy, center-crop consistency, label remap, Z-score, HD95 NaN convention) is well-formed.
- `context.md` — full chronological logbook. Useful as historical record.
- `project_review.md` — audit from 2026-04-29. Mostly historical now; all 🔴 high-priority items were resolved (FPN added, 3D test eval done, requirements updated, comparison notebook created).
- `README.md` — clean, public-facing, includes the cover image and clear setup/usage instructions.

**Some duplication** exists between these four documents (results tables appear in three of them). That is acceptable for the current scope but would become a maintenance burden if results were updated.

### 2.9 Repository and version control
**Verdict:** Healthy.

- `.gitignore` excludes `.venv*/`, `PKG - BraTS-PEDs-v1/`, `processed_dataset/`, `checkpoints/`, `*.pth`, `*.npy`, `__pycache__/`, `.ipynb_checkpoints/`, `.claude/settings.local.json`.
- Initial commit and follow-up commits (README + cover image) pushed to `https://github.com/Drastid/BraTS-PEDs-brain-tumour-segmentation` (private).
- The `.claude/` directory is partially tracked (skill file under `skills/` is currently outside the ignore).

### 2.10 Results — quantitative summary
**Verdict:** Solid for a 2D slice-based approach on paediatric BraTS.

3D test set (26 subjects, post-processed):

| Metric | U-Net | FPN | SegFormer |
|---|---|---|---|
| Dice NCR | 0.5856 | 0.5590 | **0.6752** |
| Dice ED | 0.7239 | 0.7104 | **0.7229** |
| Dice ET | 0.5398 | 0.5946 | **0.6299** |
| **Mean FG** | 0.6164 | 0.6213 | **0.6760** |
| HD95 NCR (mm) | 9.21 | 8.70 | **5.23** |
| HD95 ED (mm) | **7.45** | 7.63 | 8.15 |
| HD95 ET (mm) | 17.85 | 18.69 | **14.60** |

Key observations from the JSON files and per-subject CSV:
- SegFormer wins overall mean-FG Dice and 2 of 3 HD95 metrics (NCR and ET, the medically-critical ones).
- ED is the easiest class — large, visible on T2-FLAIR, all three models above 0.71 Dice and 8 mm HD95.
- NCR and ET have 10–20 NaN HD95 cases out of 26: many test subjects do not contain those classes at all, so HD95 is undefined. This is inherent to the dataset, not a bug.
- ET shows the largest std (≈0.43 across all models): when ET is present, prediction quality is bimodal — either very good or near-zero.

---

## 3. Recommendations, Architectural Improvements, and Refactoring

### 3.1 🔴 Priority A — Code-quality fixes (small, fast, no behavioural change)

**3.1.1 Move `import warnings` to module-level in `eval_utils.py`.**
Currently imported three times inside `compute_hd95_volume()` (cases B, C, and the bare-except). Replace with a single import at the top of the file alongside `os`, `re`, `nibabel`, etc.

**3.1.2 Centralise shared constants in `src/constants.py`.**
`NUM_CLASSES = 4`, `CLASS_NAMES`, `CROP_SIZE = 192`, `ORIG_SIZE = 240`, `N_SLICES = 155`, and the BraTS-PEDs voxel-frequency vector `_VOXEL_FREQ` are duplicated across `dataset.py`, `train_utils.py`, `eval_utils.py`, and `evaluate_3d_test.py`. Pulling them into a single module eliminates drift risk and gives a single edit point if anything ever changes (e.g. cross-domain dataset with different spatial dims).

**3.1.3 Implement `compute_iou_volume()` in `eval_utils.py`.**
The project specification (`context.md` §1.5 and `project_review.md` §5.12) lists IoU as one of the required clinical metrics, but it is not computed anywhere. The implementation is a 10-line function with the same shape as `compute_dice_volume`:

```python
def compute_iou_volume(pred, gt, smooth=1e-6):
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        pred_c = (pred == c)
        gt_c   = (gt   == c)
        inter  = float((pred_c & gt_c).sum())
        union  = float((pred_c | gt_c).sum())
        results[f"iou_{name}"] = (inter + smooth) / (union + smooth)
    return results
```

Add to `src/evaluate_3d_test.py` evaluation pipeline and to the JSON output schema.

### 3.2 🟡 Priority B — Architectural / hygiene improvements

**3.2.1 Add a `tests/` directory with unit tests for critical functions.**
At minimum:
- `tests/test_losses.py`: `DiceLoss` returns 0 on perfect predictions, returns ~1 on inverted predictions; `FocalLoss` reduces to cross-entropy at γ=0.
- `tests/test_eval_utils.py`: `compute_dice_volume` on synthetic volumes; `remove_small_components` on planted blobs; HD95 edge cases (A, B, C) return NaN.
- `tests/test_dataset.py`: `clip_outliers` clips at percentile; `zscore_normalise` produces μ≈0 σ≈1 on non-zero voxels.

These give a regression net for any future refactor (especially the constants extraction above).

**3.2.2 Document IoU + add to JSON schema.**
After implementing `compute_iou_volume`, update `evaluate_3d_test.py` to compute and serialise IoU alongside Dice and HD95. Re-run the script for all three models (it's a `<10 minute` operation since predictions are deterministic given the checkpoints).

**3.2.3 Consider deleting unused venvs.**
`.venv-1`, `.venv-2`, `.venv-3`, `.venv-4` are dead artefacts from the Python 3.14 setup pain. They are already in `.gitignore` so they do not pollute git, but they consume disk space (each .venv with PyTorch+CUDA is ~5–10 GB). Single safe deletion after confirming `.venv` works.

**3.2.4 Document the 2D-vs-3D Dice gap in the README.**
The README currently mentions "The ~20–25% Dice drop from 2D to 3D is expected for slice-based architectures with no 3D context." This is already there — good. Consider expanding with a one-sentence note that the 2D number measures *slice-level* agreement (the average over all axial slices in the test set, each scored independently), while the 3D number reconstructs the full volume per subject and scores once per subject. The two are different denominators.

### 3.3 🟢 Priority C — Optional / nice-to-have

**3.3.1 Save NIfTI predictions for clinical visual review.**
`volume_to_nifti()` and `load_subject_nifti_meta()` already exist in `eval_utils.py`. A small script (or a section in `08_comparison.ipynb`) that saves `pred.nii.gz` for the best and worst test subjects per model would let a clinician load them into ITK-SNAP or 3D Slicer.

**3.3.2 Cross-domain evaluation.**
`07_cross_domain_evaluation.ipynb` is structured but never executed because the adult BraTS dataset is not available locally. If/when the dataset becomes available, run `02_preprocessing.ipynb` with `DATA_ROOT` pointed at it, then set `NEW_DATASET_PATH` in `07` and execute. Zero-shot domain shift quantification would meaningfully strengthen the report.

**3.3.3 Reconcile `EDA_02_outputs/` vs `processing_02_outputs/`.**
There are two directories storing FPN/SegFormer training curves (`EDA_02_outputs/fpn_training_curves.png` + `processing_02_outputs/segformer_training_curves.png`). Move both under a single canonical name (e.g. `training_outputs/`) and update the notebooks. The current split is historical and confusing.

**3.3.4 Make `.claude/skills/code-reviewer.md` shareable or local-only by an explicit decision.**
The skill file (this very skill) lives under `.claude/skills/`. `.gitignore` excludes only `.claude/settings.local.json`, so the skill is *tracked* by default. If the user wants it private, add `.claude/skills/` (or `.claude/` entirely except certain files) to `.gitignore`. If shareable, the current state is correct and teammates running Claude Code on this repo will get the slash command for free.

### 3.4 Items already resolved by previous work — do not re-attempt
- ✅ FPN model (`06_train_fpn.ipynb`, `checkpoints/fpn/`)
- ✅ 3D test-set evaluation for all three models (`evaluation_outputs/test_3d_metrics_*.json`)
- ✅ Three-way comparison (`08_comparison.ipynb`, `comparison_outputs/`)
- ✅ `requirements.txt` includes `transformers` and `accelerate`
- ✅ `.gitignore` exists and is correct
- ✅ Split documented with SEED=42 in `split.json`

---

## 4. Critical Invariants (must not be violated by any future change)

These are restated from `CLAUDE.md` §8 because they are easy to break and hard to detect:

1. **Augmentation policy** — geometric transforms only. Adding any pixel-level transform corrupts the Z-score-normalised float32 input.
2. **Center-crop consistency** — `top=left=24`, `crop_size=192`, identical at training and inference. The un-crop in `predict_volume()` writes predictions back into the 240×240 canvas at the same offsets.
3. **Label remap** `seg == 4 → 3` — applied in `load_subject()`. Required for any dataset using the old BraTS ET convention.
4. **Z-score on non-zero voxels only** — background must be re-zeroed after normalisation.
5. **HD95 NaN convention** — `np.nan` for both-empty / GT-only / pred-only; downstream uses `np.nanmean` and reports NaN count separately.

---

## 5. Overall Assessment

| Aspect | Rating | Comment |
|---|---|---|
| Code architecture | ★★★★★ | Cleanly modular, single-responsibility per file |
| Documentation | ★★★★★ | Docstrings, CLAUDE.md, README, context.md all present and consistent |
| Spec adherence | ★★★★★ | All five phases done; FPN added back; test-set 3D eval done |
| Pipeline completeness | ★★★★☆ | One spec metric (IoU) missing; cross-domain blocked on data |
| Result quality | ★★★★☆ | 0.676 mean-FG Dice (3D) is strong for 2D slice-based on paediatric BraTS |
| Reproducibility | ★★★★★ | SEED=42, split.json saved, requirements.txt complete |
| Code quality | ★★★★☆ | One cosmetic defect (3× `import warnings`), some constant duplication |

**Verdict:** Project is in a *complete and shippable state* (~95%). The remaining 5% is opportunistic polish: IoU implementation, unit tests, constant centralisation, optional cross-domain execution.
