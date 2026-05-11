# CLAUDE.md

Directives and rules for Claude Code working in this repository. Keep this file concise — narrative goes in `context.md`, results go in `README.md`, plans go in `next_steps_action_plan.md`.

---

## 1. Project (one-line)

BraTS-PEDs-v1 paediatric brain tumour segmentation — 4-channel MRI → 4-class 2D slice-based segmentation comparing U-Net, FPN, SegFormer-B1.

Companion docs: `README.md` (public-facing), `context.md` (full logbook), `current_progress_review.md` (latest review), `next_steps_action_plan.md` (open work), `project_review.md` (historical audit).

---

## 2. Critical Invariants — DO NOT VIOLATE

These are correctness-critical. Breaking any of them silently corrupts results.

1. **Augmentation = geometric only.** `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `ElasticTransform`. No brightness/contrast/jitter — input is Z-score float32 with negative values.
2. **Center crop = 192×192 from 240×240 at `top=24, left=24`.** Identical at training (`train_utils.center_crop`) and inference (`eval_utils.predict_volume`). The un-crop in `predict_volume` writes back at the same offsets.
3. **Label remap `seg == 4 → 3`** applied in `dataset.load_subject()`. Required for the old BraTS ET convention.
4. **Z-score on non-zero voxels only.** Background (original zeros) is forced back to 0 after normalisation.
5. **HD95 NaN convention.** Both-empty / GT-only / pred-only → `np.nan`. Downstream uses `np.nanmean`; NaN count reported separately.
6. **Split is frozen** in `split.json` with `SEED=42` (Strategy C — ET+Quartile). 205 train / 26 val / 26 test. Do not regenerate.

---

## 3. Commands

```bash
# Install
pip install -r requirements.txt

# 3D evaluation on test set
python -m src.evaluate_3d_test --model all          # all three
python -m src.evaluate_3d_test --model unet         # one only: unet|fpn|segformer
```

Notebook execution order (only if regenerating outputs from scratch):
`01_EDA` → `02_preprocessing` (~20 min, prereq for training) → `03_train_unet` → `04_evaluation` → `05_train_segformer` → `06_train_fpn` → `06b_evaluation_fpn` → `08_comparison`. (`07_cross_domain` is template-only, awaiting adult BraTS data.)

---

## 4. Module Map (`src/`)

| File | Role |
|---|---|
| `dataset.py` | `BraTSDataset` + preprocessing primitives (`clip_outliers`, `zscore_normalise`, `load_subject`) |
| `losses.py` | `DiceLoss`, `FocalLoss`, `CombinedLoss` (returns `(total, d_loss, f_loss)`) |
| `train_utils.py` | Training/eval loop, `center_crop`, `set_encoder_trainable`, checkpoint I/O |
| `eval_utils.py` | 3D inference (`predict_volume`), `remove_small_components`, `compute_dice_volume`, `compute_hd95_volume` |
| `models.py` | `SegFormerWrapper` + 4-channel patch-embedding adaptation |
| `evaluate_3d_test.py` | CLI: 3D test-set evaluation, JSON output to `evaluation_outputs/` |

---

## 5. Training Configuration (shared by all 3 models)

- Crop 192×192 · batch 16 · Phase 1 (5 ep, frozen encoder, LR=3e-4) → Phase 2 (25 ep, full FT, encoder LR=1e-5, decoder LR=1e-4)
- Loss: `CombinedLoss` (Dice + Focal γ=2.0, weights 1:1, ignore_background)
- AdamW, weight decay 1e-4, CosineAnnealingLR per phase, AMP on CUDA, grad clip `max_norm=1.0`
- Class weights (inverse-frequency): BG≈0.0004, NCR≈0.533, ED≈0.080, ET≈0.387

Detailed results → `README.md` §Results. Per-epoch tables → `context.md`.

---

## 6. Git-ignored (do not commit)

`processed_dataset/` (~20 GB .npy), `checkpoints/` (.pth files), `PKG - BraTS-PEDs-v1/` (raw NIfTI), `.venv*/`, `__pycache__/`, `.ipynb_checkpoints/`, `.claude/settings.local.json`.

---

## 7. Operational Rules

- **Auto-commit + push** after every file change with a descriptive message. Do not wait for the user to ask.
- **Skill-driven reviews:** the `/code-reviewer` skill at `.claude/skills/code-reviewer.md` regenerates `current_progress_review.md` + `next_steps_action_plan.md` and commits them.
- **CLAUDE.md must stay short.** Push verbose content into the appropriate satellite document instead.

---

## 8. Known Open Items

See `next_steps_action_plan.md`. All code-side spec gaps closed as of 2026-05-11 (IoU, `import warnings`, constants centralisation, unit tests, NIfTI exports, output directories — done). Only remaining roadmap item is TASK 9 (cross-domain zero-shot evaluation), blocked on external adult BraTS data.
