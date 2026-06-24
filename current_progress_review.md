# Current Progress Review — BraTS-PEDs Brain Tumour Segmentation

**Reviewer:** `/code-reviewer` skill (refresh)
**Date:** 2026-05-15
**Scope:** Full codebase audit, post `0487eac` (notebooks moved into `notebooks/`)
**Supersedes:** 2026-05-11 review (everything listed as closed there *is* still closed; one new substantive finding raised here)

---

## 1. Executive Summary

The project is in a **shippable state — ~99% complete**. Every closed item from the prior review remains closed; the new audit pass confirms code structure, results, and documentation are consistent. Test suite (9 tests) **passes in 14 s on `.venv-3`** (CPU run was attempted on `.venv`, but `.venv` does not have `pytest` installed — see §4).

**One new substantive finding** — not a bug, but a **metric-reporting gap**: the Laplace-smoothed Dice and IoU return **1.0** when a class is absent in both prediction and ground truth, and these `1.0`s are mixed into the per-class mean without being flagged. HD95 correctly returns `NaN` for the same case and is excluded by `np.nanmean`. The asymmetry means **per-class Dice/IoU means in the test set are inflated by 6–14 degenerate `1.0` values per class** (see §3.2). The relative ranking between models is unaffected (same convention applied uniformly), but the absolute numbers overstate per-class performance on small classes. Worth surfacing in the JSON output + README.

**Winner (unchanged):** SegFormer-B1 — mean foreground 3D Dice **0.6760**, mean foreground 3D IoU **0.6047**.

**Outstanding roadmap items:**
- 🔴 TASK 9 — Cross-domain zero-shot evaluation (blocked on adult BraTS data).
- 🟡 TASK 11 (new) — Surface Dice/IoU degenerate-case counts; expose `nanmean`-over-present-class variants.
- 🟢 TASK 12 (new) — Either delete `notebooks/06b_evaluation_fpn.ipynb` (currently 0 executed cells, superseded by `python -m src.evaluate_3d_test --model fpn`) or execute it once and commit the outputs.

---

## 2. Component-by-Component Review

### 2.1 `src/constants.py`
**Verdict:** Clean. Single source of truth for `NUM_CLASSES`, `CLASS_NAMES`, `CROP_SIZE`, `ORIG_SIZE`, `N_SLICES`, `VOXEL_FREQ`. Re-exported through `dataset.py` (`# noqa: F401`) so existing callers keep working.

### 2.2 `src/dataset.py`
**Verdict:** Excellent. Unchanged from prior reviews.

- Preprocessing pipeline (clip P99.5 → Z-score → background re-zeroed → label remap 4→3) implements Critical Invariants #3 and #4 cleanly.
- `BraTSDataset.__getitem__` does the HWC↔CHW transpose around augmentation correctly.
- `clip_outliers`, `zscore_normalise` covered by `tests/test_dataset.py` (2 tests).

### 2.3 `src/losses.py`
**Verdict:** Excellent.

- `DiceLoss`: softmax + one-hot + per-class weighted mean. Optional `ignore_background` excludes class 0 from the mean (intentional — class 0 trivially scores ≈1.0 and would dominate the metric).
- `FocalLoss`: numerically stable via `log_softmax + gather`. `targets.clamp(min=0)` defends against `ignore_index=-100` poisoning the `gather` call; `ignore_index >= 0` is masked off post-gather. Correct.
- `CombinedLoss.forward` returns the `(total, d_loss, f_loss)` tuple expected by `train_one_epoch()`.
- 3 unit tests in `tests/test_losses.py` cover perfect prediction, inverted prediction, and the γ=0 ⇒ cross-entropy equivalence.

### 2.4 `src/train_utils.py`
**Verdict:** Strong.

- Augmentation policy (`get_augmentation`) is geometric-only — implements Critical Invariant #1. `ShiftScaleRotate(border_mode=0)` fills out-of-frame regions with `0`, consistent with the Z-score convention (`0` ≡ original background).
- `center_crop(image, mask, 192)` uses `top=(H-crop)//2`, `left=(W-crop)//2` — identical formula as `eval_utils._CROP_TOP/_CROP_LEFT`, satisfying Critical Invariant #2.
- `train_one_epoch` runs AMP + grad-clip(max_norm=1.0) correctly; the `device_type=str(device).split(":")[0]` trick correctly extracts `cuda` from `cuda:0`.
- `get_class_weights()` consumes `VOXEL_FREQ` from `constants.py` and produces a tensor that sums to 1.

### 2.5 `src/eval_utils.py`
**Verdict:** Most-improved module of the prior review; still solid.

- `predict_volume` does the slice-by-slice forward + un-crop back to ORIG_SIZE, with auto-detection of `(orig_size, n_slices)` when `None` — important for cross-domain runs.
- `remove_small_components` uses 26-connectivity (`generate_binary_structure(3, 3)`) and processes one foreground class at a time. The "confetti effect" is mitigated, not eliminated — small *true* lesions <50 voxels are also removed. This is documented and is a deliberate trade-off.
- `compute_hd95_volume` correctly implements the **Case A / Case B / Case C** NaN convention. RuntimeWarnings are emitted for B/C only (A is silent — both empty is expected, not an error). `connectivity=1` (medpy's face-connectivity for surface extraction) is the standard BraTS choice; do **not** confuse this with the 26-connectivity used elsewhere — they govern different operations.
- `compute_iou_volume` mirrors `compute_dice_volume` exactly (Laplace smoothing, per-class results dict). **See §3.2 for the degenerate-case finding.**

### 2.6 `src/models.py`
**Verdict:** Unchanged. The 4-channel patch-embedding adaptation (preserve RGB weights, init channel 4 = mean(RGB)) is elegant and reproducible.

### 2.7 `src/evaluate_3d_test.py`
**Verdict:** Solid; computes Dice + HD95 + IoU per subject and aggregates.

- Lazy import of `src.models.get_segformer` inside `load_segformer_model` lets you evaluate U-Net / FPN without pulling `transformers`. Nice touch.
- `nan_to_none` helper handles JSON serialisation of NaN.
- **Minor reporting gap (§3.2):** the summary does report `hd95_nan_count` per class, but **does not report a `dice_degenerate_count` or `iou_degenerate_count`**. A subject where the class is absent in both pred and GT contributes a silent `1.0` to the Dice mean.

### 2.8 `src/export_nifti_predictions.py`
**Verdict:** Clean reuse of `evaluate_3d_test` constants + loaders. Self-documenting `pick_best_worst` heuristic (sort by mean fg Dice across NCR/ED/ET). Output: 6 `.nii.gz` files under `evaluation_outputs/nifti_predictions/{unet,fpn,segformer}/` + a `README.md` explaining how to overlay them in 3D Slicer / ITK-SNAP. Good clinical artefact.

### 2.9 `tests/`
**Verdict:** Modest regression net; 9 tests passing in 14 s. **Confirmed locally on `.venv-3`** (`.venv` does not have `pytest` installed — install if you want to run tests there too).

Coverage map:

| Tested | Untested (no regression net today) |
|---|---|
| `DiceLoss` (perfect, inverted) | `CombinedLoss.forward` tuple unpacking |
| `FocalLoss` (γ=0 ⇒ CE) | `FocalLoss.alpha` weighting path |
| `compute_dice_volume` (identity) | `compute_dice_volume` degenerate case (both empty ⇒ 1.0) |
| `compute_iou_volume` (identity) | `compute_iou_volume` degenerate case |
| `remove_small_components` (10/60/1000 mix) | `predict_volume` shape contract |
| `compute_hd95_volume` (A/B/C + normal) | `get_segformer` 4-ch adaptation |
| `clip_outliers` (1e6 outlier) | `MetricTracker` running mean |
| `zscore_normalise` (bg=0, μ=0, σ=1) | `center_crop`, `get_class_weights`, `set_encoder_trainable` |

The most valuable additions would test (a) the degenerate-case behaviour of Dice/IoU explicitly (locks in the convention or motivates a switch to NaN), and (b) `get_segformer` adaptation (sanity check on `proj.weight.shape == [C_out, 4, k, k]`).

### 2.10 Notebooks
**Verdict:** Two are stubs; rest are fine.

| Notebook | Cells | Code cells executed | State |
|---|---|---|---|
| `01_EDA.ipynb` | 20 | 10/10 | ✅ executed |
| `02_preprocessing.ipynb` | 27 | 15/15 | ✅ executed |
| `03_train_unet.ipynb` | 27 | 14/14 | ✅ executed |
| `04_evaluation.ipynb` | 19 | 11/11 | ✅ executed (val set, U-Net only — historical) |
| `05_train_segformer.ipynb` | 29 | 15/15 | ✅ executed |
| `06_train_fpn.ipynb` | 25 | 12/12 | ✅ executed |
| `06b_evaluation_fpn.ipynb` | 7 | **0** | 🟡 stub — superseded by `evaluate_3d_test.py` |
| `07_cross_domain_evaluation.ipynb` | 25 | **0** | 🔴 blocked on adult BraTS data |
| `08_comparison.ipynb` | 21 | 10/10 | ✅ executed |

`06b_evaluation_fpn.ipynb` exists as a thin wrapper around `subprocess.run([python_exe, "-m", "src.evaluate_3d_test", ...])` plus a results-display section. Since the CLI script is now canonical (and listed first in `README.md` usage), the notebook adds nothing — either run it once and commit the outputs, or delete it.

### 2.11 Output directories
**Verdict:** Clean; matches the prior review's reconciliation.

| Directory | Contents |
|---|---|
| `EDA_01_outputs/` | 4 EDA plots from `notebooks/01_EDA.ipynb` |
| `processing_02_outputs/` | 5 `preproc_*.png` (preprocessing QC) |
| `training_outputs/` | 8 PNGs (training curves, val predictions, failure analysis) |
| `evaluation_outputs/` | 3 JSON test-metric files + `nifti_predictions/` (6 `.nii.gz` + README) |
| `comparison_outputs/` | 5 PNG + 2 CSV from `notebooks/08_comparison.ipynb` |

### 2.12 `requirements.txt` & environment
**Verdict:** Functional but unpinned.

- All packages required by current code paths are listed (`pytest` was added; `transformers`/`accelerate` are present for SegFormer).
- **No versions pinned.** `transformers` 5.6.2 is a major-version release with breaking changes vs 4.x; running `pip install -r requirements.txt` against a fresh interpreter today may install a version that no longer exposes `SegformerForSemanticSegmentation` with the same API. Pinning is cheap insurance.
- Five venvs on disk (`.venv`, `.venv-1`–`.venv-4`); per memory `project_venvs.md`, **`.venv-3`** is the primary GPU env, `.venv-4` the backup, `.venv` CPU-only, and `.venv-1`/`.venv-2` are dead. Confirmed: tests run cleanly in `.venv-3`.

### 2.13 Documentation
**Verdict:** Up to date.

- `README.md`: reports IoU + Dice + HD95; 2D-vs-3D denominator note present.
- `context.md`: complete logbook through 2026-04-30.
- `CLAUDE.md`: §8 already reads "All code-side spec gaps closed as of 2026-05-11" — accurate. The new TASK 11 (Dice degenerate-case reporting) does not warrant a CLAUDE.md update — it is a reporting enhancement, not an invariant or workflow change.
- `project_review.md`: 2026-04-29 historical audit; explicitly marked superseded.

---

## 3. Results & one new substantive finding

### 3.1 3D test set — headline numbers (n=26, post-processed)

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

### 3.2 🟡 NEW finding — Dice/IoU degenerate cases silently inflate per-class means

**What the code does.** `compute_dice_volume` and `compute_iou_volume` apply Laplace smoothing (`ε = 1e-6`). When both prediction and ground truth are empty for a class (`Case A` in HD95 parlance), this yields `(0 + ε) / (0 + ε) = 1.0` — the function returns *perfect agreement* for a class that does not appear anywhere. This is documented in the docstring (*"Returns 1.0 (perfect agreement) in that degenerate case."*) but the per-class **mean** in the JSON summary and in `README.md` does not flag how many subjects contributed a degenerate `1.0`.

**Why it matters.** Two of the three foreground classes (NCR, ET) are absent in many test subjects. Empirically (count of subjects where `dice = 1.0` *and* `hd95 = None`, i.e. both-empty cases):

| Model | NCR Dice = 1.0 | ET Dice = 1.0 | (HD95 NaN total — for context) |
|---|---|---|---|
| U-Net | 6 / 26 | 11 / 26 | NCR: 10, ET: 17 |
| FPN | 7 / 26 | 13 / 26 | NCR: 12, ET: 20 |
| SegFormer | 7 / 26 | 14 / 26 | NCR: 10, ET: 20 |

The HD95 NaN total is higher than the Dice-1.0 count because HD95 NaN also covers **Case B** (GT-only, complete miss → Dice ≈ 0, hd95 = NaN) and **Case C** (pred-only, hallucination → Dice ≈ 0, hd95 = NaN). Those contribute realistic low Dice values; only **Case A** contributes the inflating `1.0`.

**Effect on the headline.** Re-computing per-class mean Dice over the 26 − (Case-A count) subjects:

| Model | NCR Dice (reported) | NCR Dice (excl. Case A) | ET Dice (reported) | ET Dice (excl. Case A) |
|---|---|---|---|---|
| U-Net | 0.5856 | ≈ 0.4613 | 0.5398 | ≈ 0.2023 |
| FPN | 0.5590 | ≈ 0.4006 | 0.5946 | ≈ 0.0641 |
| SegFormer | 0.6752 | ≈ 0.5555 | 0.6299 | ≈ 0.1982 |

(*ED stays unchanged because Case-A count = 0 for ED — every subject has some oedema.*)

**Three observations:**
1. **Model ranking is unaffected** — SegFormer remains the best on both reported and Case-A-excluded means for every class. The comparison is internally consistent.
2. **Absolute ET numbers are dramatically lower** when Case A is excluded (U-Net 0.54 → 0.20). The "raw" 3D Dice ET on subjects where ET actually exists is the more honest single-number summary for a clinical reader, since "predict no ET on subjects with no ET" is trivial.
3. **Convention is defensible but should be surfaced.** Many BraTS papers report `Dice = 1.0` for empty/empty pairs; others use NaN + nanmean. The current code chose the former *via the smooth term*, which is fine — but it must be flagged in the JSON output and the README, mirroring the way `hd95_nan_count` is already flagged.

**Suggested remediation** (does not affect existing checkpoints/training, only reporting): see TASK 11 in `next_steps_action_plan.md`.

### 3.3 Other observations from the audit

- **`evaluate_3d_test.py` docstring** still names *"U-Net" or "SegFormer"* in the `model_name` description (line 167). It actually supports all three including FPN. Cosmetic — fix in passing.
- **`README.md` Project Structure tree** lists `src/evaluate_3d_test.py` but **does not list `src/export_nifti_predictions.py` or `src/constants.py`**. Both are committed, used, and tested. Worth adding the two missing entries.
- **`notebooks/04_evaluation.ipynb`** is the old val-set-only U-Net eval. It is historical (superseded by `evaluate_3d_test --model unet` on the **test** set) but still useful as a sanity-check on the val pipeline. Leave as is.
- **No CI.** The 9 pytest tests will only run when someone manually invokes `pytest`. Cheap optional polish: GitHub Actions workflow that runs `pytest -q` on push (Python ≥ 3.10, `pip install -r requirements.txt`). Probably out of scope, but mentioned in the action plan as optional.

---

## 4. Verifications performed in this audit

| Check | Result |
|---|---|
| `pytest -q` on `.venv-3` | ✅ 9 passed in 14.18 s |
| `pytest -q` on `.venv` | ❌ `No module named pytest` (CPU env lacks pytest — install if needed) |
| `git status` | Clean working tree (only `.claude/` untracked, gitignored) |
| `compute_dice_volume`/`compute_iou_volume` degenerate-case empirics | 6–14 subjects per class per model (§3.2) — confirmed by reading the live JSONs in `evaluation_outputs/` |
| Critical invariants still hold (CLAUDE.md §2 #1–#6) | ✅ all six |
| Notebook execution states | 7/9 executed; `06b_evaluation_fpn` stub, `07_cross_domain_evaluation` blocked (§2.10) |

---

## 5. Critical Invariants — re-verified, do not violate

(Unchanged from `CLAUDE.md` §2. Re-listed here for completeness — every reviewer pass must confirm these still hold.)

1. **Augmentation = geometric only** (HFlip, Rotate90, ShiftScaleRotate, ElasticTransform). No brightness/contrast/jitter — input is Z-score float32 with negative values.
2. **Center crop 192×192 at `top=24, left=24`**, identical at training (`train_utils.center_crop`) and inference (`eval_utils.predict_volume`). The un-crop in `predict_volume` writes back at the same offsets.
3. **Label remap `seg == 4 → 3`** applied in `dataset.load_subject()`.
4. **Z-score on non-zero voxels only**; background re-zeroed after normalisation.
5. **HD95 NaN convention** — both-empty / GT-only / pred-only → `np.nan`; downstream uses `np.nanmean`.
6. **Split frozen** in `split.json` with `SEED=42` (205 train / 26 val / 26 test).

---

## 6. Overall Assessment

| Aspect | Rating | Change vs 2026-05-11 |
|---|---|---|
| Code architecture | ★★★★★ | unchanged |
| Documentation | ★★★★★ | unchanged |
| Spec adherence | ★★★★★ | unchanged |
| Pipeline completeness | ★★★★★ | unchanged |
| Result quality | ★★★★☆ | unchanged |
| Reproducibility | ★★★★☆ | ↓ ½ — `requirements.txt` is still unpinned (was already an open issue, now noted as TASK 13 / optional) |
| Code quality | ★★★★★ | unchanged |
| Metric reporting honesty | ★★★★☆ | NEW dimension — Dice/IoU degenerate-case (§3.2) merits a 🟡 task to surface the count in JSON + README |
| Testing | ★★★★☆ | unchanged — gaps remain (see §2.9 table) |
| Clinical artefacts | ★★★★★ | unchanged |

**Verdict.** Project is complete and shippable. The new finding (§3.2) is a reporting enhancement that strengthens the credibility of the headline numbers; it does **not** invalidate any committed result and does **not** block shipping. All open work items are now consolidated in `next_steps_action_plan.md`.
