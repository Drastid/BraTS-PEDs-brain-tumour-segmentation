# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Project Overview

**Course**: AI in Medicine — Politecnico di Torino, MSc (A.Y. 2025-2026)

**Clinical objective**: Automated semantic segmentation of paediatric brain gliomas (GTV delineation) from multi-parametric MRI, targeting neuroncologists and neurosurgeons. Goal: reduce inter-operator variability and segmentation time.

**Dataset**: BraTS-PEDs-v1 — 257 training subjects (NIfTI, 4 MRI sequences, 240×240×155, 1mm isotropic).

**Task**: 2D slice-based multi-class segmentation → 4 classes: `{0=BG, 1=NCR, 2=ED/SNFH, 3=ET}`.

**Input tensor**: `[B, 4, H, W]` — four co-registered modalities: T1c, T1n, T2f, T2w.

---

## 2. Project Status (as of 2026-05-11)

All five development phases are **complete**. The project is fully functional.

| Phase | Status | Key output |
|---|---|---|
| EDA & NIfTI parsing | ✅ | `EDA_01_outputs/` (4 plots) |
| 3D→2D preprocessing & split | ✅ | `processed_dataset/` (39,538 `.npy` pairs), `split.json` |
| U-Net/ResNet34 training | ✅ | `checkpoints/unet/best.pth` (epoch 18, 293 MB) |
| FPN/ResNet34 training | ✅ | `checkpoints/fpn/best.pth` (epoch 15) |
| SegFormer-B1 training | ✅ | `checkpoints/segformer/best.pth` (epoch 14, 164 MB) |
| 3D clinical evaluation (test set) | ✅ | `evaluation_outputs/test_3d_metrics_*.json` |
| Three-way comparison | ✅ | `comparison_outputs/` (CSVs + plots) |
| Cross-domain evaluation | 🔶 | `07_cross_domain_evaluation.ipynb` (structured, not executed — needs adult BraTS dataset) |

---

## 3. Commands

```bash
# Install dependencies
pip install -r requirements.txt

# 3D evaluation on test set — all models
python -m src.evaluate_3d_test --model all

# 3D evaluation — single model
python -m src.evaluate_3d_test --model unet
python -m src.evaluate_3d_test --model fpn
python -m src.evaluate_3d_test --model segformer
```

**Notebook execution order**: `01_EDA` → `02_preprocessing` → `03_train_unet` → `04_evaluation` → `05_train_segformer` → `06_train_fpn` → `06b_evaluation_fpn` → `07_cross_domain_evaluation` → `08_comparison`.

`02_preprocessing.ipynb` must run first — it extracts all `.npy` slices into `processed_dataset/` (prerequisite for all training notebooks). Extraction takes ~20 minutes.

---

## 4. Repository Structure

```
Project/
├── src/                        # Python source modules
│   ├── dataset.py              # BraTSDataset + preprocessing utilities
│   ├── losses.py               # DiceLoss, FocalLoss, CombinedLoss
│   ├── train_utils.py          # Training/eval loop, center_crop, checkpointing
│   ├── eval_utils.py           # 3D reconstruction, HD95, post-processing
│   ├── models.py               # SegFormerWrapper (drop-in compatibility layer)
│   └── evaluate_3d_test.py     # CLI script for 3D test-set evaluation
├── 01_EDA.ipynb                # Exploratory data analysis
├── 02_preprocessing.ipynb      # 3D→2D extraction, split, QC
├── 03_train_unet.ipynb         # U-Net/ResNet34 training
├── 04_evaluation.ipynb         # 3D evaluation (val set, U-Net only)
├── 05_train_segformer.ipynb    # SegFormer-B1 training
├── 06_train_fpn.ipynb          # FPN/ResNet34 training
├── 06b_evaluation_fpn.ipynb    # 3D evaluation (val set, FPN)
├── 07_cross_domain_evaluation.ipynb  # Cross-domain zero-shot eval (template)
├── 08_comparison.ipynb         # Three-way model comparison
├── split.json                  # Train/val/test subject IDs (SEED=42)
├── requirements.txt            # Python dependencies
├── context.md                  # Full project logbook (specifica + execution log)
├── project_review.md           # Code audit report (2026-04-29)
│
├── checkpoints/                # [git-ignored] Model weights (.pth)
│   ├── unet/{best,last}.pth + history.json
│   ├── fpn/{best,last}.pth + history.json
│   └── segformer/{best,last}.pth + history.json
├── processed_dataset/          # [git-ignored] Extracted 2D .npy slices
│   ├── train/images/ + masks/  # 31,519 slices
│   ├── val/images/ + masks/    # 4,009 slices
│   └── test/images/ + masks/   # 4,010 slices
├── PKG - BraTS-PEDs-v1/       # [git-ignored] Raw NIfTI data
├── .venv*/                     # [git-ignored] Virtual environments
│
├── EDA_01_outputs/             # EDA plots (tracked)
├── EDA_02_outputs/             # FPN training curves (tracked)
├── processing_02_outputs/      # Preprocessing QC + training curves (tracked)
├── evaluation_outputs/         # 3D test metrics JSON (tracked)
└── comparison_outputs/         # Comparison CSVs + plots (tracked)
```

`.npy` slice naming: `<subject_id>_slice<idx>.npy` — images shape `[4, 240, 240]` float32, masks shape `[240, 240]` int8.

---

## 5. Architecture

### 5.1 `src/dataset.py`

`BraTSDataset` — lightweight PyTorch Dataset backed by pre-extracted `.npy` files. `__getitem__` does mmap-backed disk reads (~38.6 ms/batch vs 5000–15000 ms for live NIfTI loading).

Preprocessing pipeline (applied at NIfTI extraction time):
1. Load NIfTI → float32
2. `clip_outliers()` — clip non-zero voxels at P99.5 (eliminates scanner artifacts)
3. `zscore_normalise()` — Z-score on **non-zero (brain) voxels only**; background forced back to 0
4. Label remap: `seg == 4 → 3` (old BraTS ET convention unification)

### 5.2 `src/losses.py`

- `DiceLoss` — soft multi-class Dice from logits, `ignore_background=True` (excludes class 0), Laplace smoothing ε=1e-6
- `FocalLoss` — multi-class focal, γ=2.0, per-class α weights, numerically stable via `log_softmax + gather`
- `CombinedLoss` — `dice_weight × DiceLoss + focal_weight × FocalLoss`; `forward()` returns `(total, d_loss, f_loss)` tuple for independent logging

### 5.3 `src/train_utils.py`

| Symbol | Purpose |
|---|---|
| `get_augmentation(p)` | Geometric-only: HFlip, Rotate90, ShiftScaleRotate, ElasticTransform |
| `get_class_weights()` | Inverse-frequency weights (BG=0.000354, NCR=0.5332, ED=0.0798, ET=0.3867) |
| `MetricTracker` | Running-mean accumulator for named scalars |
| `center_crop(image, mask, 192)` | Deterministic 192×192 center crop (`top=24, left=24` from 240×240) |
| `set_encoder_trainable(model, bool)` | Freeze/unfreeze `model.encoder` for two-phase training |
| `train_one_epoch()` | AMP, gradient clipping (max_norm=1.0), MetricTracker |
| `evaluate()` | `@torch.no_grad()` validation loop |
| `save_checkpoint()` / `load_checkpoint()` | Full state serialisation (model + optimizer + scheduler + scaler + epoch) |

### 5.4 `src/models.py`

`SegFormerWrapper` makes HuggingFace SegFormer a drop-in replacement for SMP models in the shared training loop:
- `.encoder` → exposes `model.segformer.encoder` for `set_encoder_trainable()`
- `.decode_head` → exposes `model.decode_head` for differential LR
- `forward()` → upsamples logits from `[B, C, H/4, W/4]` to `[B, C, H, W]` via bilinear interpolation

4-channel adaptation: patch embedding expanded 3→4 channels; RGB weights preserved, 4th channel initialised as `mean(RGB)`.

`get_segformer(model_checkpoint="nvidia/mit-b1", num_classes=4)` factory function.

### 5.5 `src/eval_utils.py`

| Function | Purpose |
|---|---|
| `get_subject_ids(data_dir)` | List unique subject IDs from slice filenames |
| `infer_dataset_shape(data_dir, subject_id)` | Auto-detect `orig_size` and `n_slices` for cross-domain use |
| `predict_volume(model, data_dir, subject_id, device)` | Slice-by-slice 3D inference with center-crop + un-crop consistency |
| `remove_small_components(volume, min_voxels=50)` | Post-process: remove blobs <50 voxels (26-connectivity, per-class) |
| `compute_dice_volume(pred, gt)` | Per-class 3D Dice |
| `compute_hd95_volume(pred, gt, voxel_spacing)` | Per-class HD95 via medpy; NaN on degenerate cases |

### 5.6 `src/evaluate_3d_test.py`

Standalone CLI for 3D test-set evaluation. Loads model, runs `predict_volume` → `remove_small_components` → `compute_dice_volume` + `compute_hd95_volume` for all 26 test subjects. Saves results as JSON.

---

## 6. Training Configuration (identical for all three models)

| Parameter | Value |
|---|---|
| Crop size | 192×192 (center crop from 240×240) |
| Batch size | 16 |
| Phase 1 (frozen encoder) | 5 epochs, LR=3e-4, CosineAnnealingLR |
| Phase 2 (full fine-tuning) | 25 epochs, encoder LR=1e-5 / decoder LR=1e-4, CosineAnnealingLR |
| Weight decay | 1e-4 |
| Loss | CombinedLoss (Dice+Focal, weights 1:1, γ=2.0, ignore_background=True) |
| AMP | Enabled on CUDA |
| Gradient clip | max_norm=1.0 |
| GPU | NVIDIA GeForce RTX 3080 Laptop (8 GB VRAM) |

---

## 7. Model Architectures & Results

### 7.1 U-Net/ResNet34 (`smp.Unet`)

- 24,439,940 total params (encoder 21,287,808 + decoder 3,152,132)
- Best val fg Dice: **0.8385** (epoch 18) — characteristic "click" loss drop at epoch 15 (0.87→0.40)
- Checkpoint: `checkpoints/unet/best.pth`

### 7.2 FPN/ResNet34 (`smp.FPN`)

- ~22,052,164 total params (same encoder as U-Net, lighter decoder ~764K)
- Best val fg Dice: **0.8461** (epoch 15) — smooth convergence (no "click")
- Checkpoint: `checkpoints/fpn/best.pth`

### 7.3 SegFormer-B1 (`nvidia/mit-b1`, wrapped by `SegFormerWrapper`)

- ~13,681,412 total params (44% fewer than U-Net)
- Best val fg Dice: **0.8554** (epoch 14)
- Checkpoint: `checkpoints/segformer/best.pth`

### 7.4 Final Results — 3D Test Set (26 subjects, post-processed)

| Metric | U-Net/ResNet34 | FPN/ResNet34 | SegFormer-B1 |
|---|---|---|---|
| dice_NCR (3D) | 0.5856 | 0.5590 | **0.6752** |
| dice_ED (3D) | 0.7239 | 0.7104 | 0.7229 |
| dice_ET (3D) | 0.5398 | 0.5946 | **0.6299** |
| **mean_fg_dice (3D)** | 0.6164 | 0.6213 | **0.6760** |
| HD95_NCR (mm) | 9.21 | 8.70 | **5.23** |
| HD95_ED (mm) | **7.45** | 7.63 | 8.15 |
| HD95_ET (mm) | 17.85 | 18.69 | **14.60** |

**Key findings**: SegFormer-B1 is best overall (+9.7% 3D fg Dice vs U-Net, +20.8% NCR Dice relative) despite 44% fewer parameters. 2D→3D performance gap is ~20-25% for all models (expected for 2D-based architectures). ET is the hardest class (17-20/26 NaN HD95 cases due to ET absence in many subjects).

---

## 8. Critical Constraints — DO NOT VIOLATE

**Augmentation policy**: Only geometric transforms permitted (`HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `ElasticTransform`). Pixel-level transforms (brightness, contrast, color jitter) are **forbidden** — inputs are Z-score normalised float32 with **negative values**. Adding pixel transforms would corrupt the normalisation.

**Center crop consistency**: Training and 3D evaluation both apply the same 192×192 center crop (`top=24, left=24`). `predict_volume()` un-crops predictions back into the 240×240 canvas before assembling volumes. Changing `CROP_SIZE` in one place without updating the other will silently produce wrong 3D reconstructions.

**Label remap**: `seg == 4 → 3` (old BraTS ET convention). Applied in `load_subject()` and must be preserved for all datasets using the old convention.

**Z-score normalisation**: Applied on non-zero (brain) voxels only; background voxels (original zeros) are forced back to 0 after normalisation. Never apply standard whole-volume normalisation.

**HD95 NaN convention**: `compute_hd95_volume()` returns `np.nan` for three degenerate cases (both empty, complete miss, hallucination). All aggregations must use `np.nanmean`. NaN counts are tracked separately in the evaluation script.

---

## 9. Data Split

- **Strategy C (ET+Quartile)**, SEED=42: ensures balanced ET presence rate across splits
- Train: 205 subjects | Val: 26 subjects | Test: 26 subjects
- Saved in `split.json` — do not re-run the split with a different seed

---

## 10. Known Issues / Future Work

- `import warnings` is repeated 3× inside `compute_hd95_volume()` body — should be moved to module-level imports (minor, non-functional issue)
- `NUM_CLASSES` and `CLASS_NAMES` are duplicated between `train_utils.py` and `eval_utils.py` — consider centralising in a `src/constants.py`
- IoU metric is not implemented — mentioned in project specification but missing from `eval_utils.py`
- Cross-domain evaluation (`07_cross_domain_evaluation.ipynb`) requires an adult BraTS dataset (e.g. BraTS2023-GLI); the notebook is ready with a `NEW_DATASET_PATH` placeholder
- NIfTI 3D visualisations (ITK-SNAP / 3D Slicer) are not included — `volume_to_nifti()` and `load_subject_nifti_meta()` in `eval_utils.py` are already implemented for this
- 5 virtual environment directories exist (`.venv` through `.venv-4`) — only `.venv` is active; the others are legacy artifacts from Python version compatibility issues during setup

---

## 11. Environment

- Python 3.14.3, PyTorch 2.11.0+cu128, CUDA GPU (RTX 3080 Laptop 8GB)
- Key packages: `segmentation-models-pytorch==0.5.0`, `transformers==5.6.2`, `albumentations==2.0.8`, `medpy==0.5.2`, `nibabel==5.4.2`
- Active venv: `.venv`
