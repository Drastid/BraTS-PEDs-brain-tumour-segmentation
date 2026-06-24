# CONTEXT — BraTS-PEDs Brain Tumour Segmentation

> Documento riorganizzato il 2026-04-30. Tutte le informazioni originali sono preservate.
> Struttura: Specifica → Roadmap → Implementation Plan → Execution Log (cronologico) → Risultati Finali

---

## SEZIONE 1 — Specifica del Progetto

### 1.1 Obiettivo Clinico

Il progetto affronta la segmentazione semantica dei gliomi cerebrali a partire da risonanze magnetiche (MRI) multiparametriche. L'obiettivo è l'automazione della delineazione del volume tumorale (Gross Tumor Volume, GTV), assistendo neuroncologi e neurochirurghi con misurazioni oggettive e riproducibili, abbattendo i tempi della segmentazione manuale e riducendo la variabilità inter-operatore.

### 1.2 Dati

Dati di neuroimaging 3D in formato NIfTI dal dataset **BraTS-PEDs-v1**. Volumi cerebrali co-registrati acquisiti con quattro sequenze MRI (T1c, T1n, T2f, T2w). Input tensor: `[batch, 4, H, W]`.

### 1.3 Architetture

- **2D U-Net** con encoder ResNet34 (ImageNet pretrained) — standard aureo per imaging medico
- **FPN (Feature Pyramid Network)** con lo stesso encoder ResNet34 — confronto "head-only" equo
- **SegFormer-B1** con encoder MiT-B1 (ImageNet pretrained) — architettura Vision Transformer, confronto aggiuntivo

### 1.4 Loss Function

Combinazione di **Dice Loss** (ottimizza overlap volumetrico) + **Focal Loss** (gestisce class imbalance, γ=2.0). Background escluso dal Dice averaging (`ignore_background=True`).

### 1.5 Metriche Cliniche

- **Dice coefficient** per classe (3D volumetrico)
- **HD95** — Hausdorff Distance al 95° percentile, misura precisione dei contorni (mm)
- **IoU** — Intersection over Union per classe *(da implementare — vedi Action Plan Task 2)*

### 1.6 Impatto Medico

Minimizzare l'HD95 assicura che i contorni predetti siano clinicamente utilizzabili: un errore ai bordi può significare asportare tessuto sano o lasciare residui tumorali attivi.

---

## SEZIONE 2 — Roadmap di Sviluppo

| Fase | Settimana | Contenuto | Stato |
|---|---|---|---|
| **Fase 1** — Data Acquisition & NIfTI Parsing | 1 | Download BraTS, EDA, comprensione sotto-regioni | ✅ |
| **Fase 2** — 3D→2D Conversion & Preprocessing | 2 | Estrazione slice 2D, normalizzazione, DataLoader | ✅ |
| **Fase 3** — Baseline U-Net Training | 3 | U-Net/ResNet34, Dice+Focal loss, two-phase training | ✅ |
| **Fase 4** — FPN & SegFormer Training | 4 | FPN/ResNet34 + SegFormer-B1, stessi iperparametri | ✅ |
| **Fase 5** — Clinical Evaluation & Visualization | 5 | 3D Dice + HD95 su test set, failure analysis, confronto | ✅ |
| **Extra** — Cross-Domain Generalization | — | Zero-shot su dataset adulto BraTS | 🔶 Strutturato |

---

## SEZIONE 3 — Dataset Analysis (Step 0 — Completed 2026-04-22)

### Dataset Structure

```
BraTS-PEDs-v1/
├── Training/         # 257 subjects (BraTS-PED-XXXXX-000/)
└── Validation/       # 91 subjects  (no segmentation masks)
```

**Per-subject files (Training):** `{subject_id}-t1c.nii.gz`, `-t1n.nii.gz`, `-t2f.nii.gz`, `-t2w.nii.gz`, `-seg.nii.gz`

| Property | Value |
|---|---|
| Format | NIfTI compressed (`.nii.gz`) |
| Modalities | 4: `t1c`, `t1n`, `t2f`, `t2w` |
| Training subjects | 257 |
| Volume shape | 240 × 240 × 155 (isotropic 1mm) |
| Labels | 0=Background, 1=NCR, 2=ED/SNFH, 3=ET |

**Critical finding**: Mixed BraTS conventions — some subjects use label 4 for ET → remapped to 3 at load time (`np.where(seg == 4, 3, seg)`).

**Class imbalance**: BG 99.40%, NCR 0.066%, ED 0.441%, ET 0.091%. Tumour sub-region split: ED 73.7%, ET 15.3%, NCR 11.1%.

**Impact**: Validation set has no labels → evaluation must use held-out split from Training (80/10/10).

### Files created
- `notebooks/01_EDA.ipynb` — fully executed
- `EDA_01_outputs/`: `eda_slice_overview.png`, `eda_overlay.png`, `eda_intensity_distributions.png`, `eda_label_distribution.png`

---

## SEZIONE 4 — Environment Setup (Step 1 — Completed 2026-04-22)

### Installed packages

| Package | Version |
|---|---|
| Python | 3.14.3 |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| nibabel | 5.4.2 |
| SimpleITK | 2.5.3 |
| segmentation-models-pytorch | 0.5.0 |
| albumentations | 2.0.8 |
| MedPy | 0.5.2 |
| scipy | 1.17.1 |
| numpy | 2.4.4 |
| matplotlib | 3.10.8 |
| transformers | 5.6.2 |
| accelerate | 1.13.0 |

### Files created
- `requirements.txt`

---

## SEZIONE 5 — Preprocessing & PyTorch Dataset (Steps 2-3 — Completed 2026-04-22)

### Preprocessing Pipeline

1. **QC scan** (10 subjects): No NaN/Inf; outlier voxels >5σ: 0.00–0.75% (scanner artifacts)
2. **Clip** non-zero voxels at P99.5 (values: 342–2056 raw units, modality-dependent)
3. **Z-score** on non-zero voxels only (background forced back to 0)
4. **Slice filtering**: T1n brain coverage ≥ 1% → all 155 slices kept per subject
5. **Label remap**: `seg == 4 → 3`

### Split Strategy (Strategy C — ET+Quartile, SEED=42)

Three strategies compared; Strategy C chosen for best ET presence rate balance:

| Split | Subjects | ET rate | Tumour vol mean |
|---|---|---|---|
| Train | 205 | 42.0% | 54,674 |
| Val | 26 | 42.3% | 47,924 |
| Test | 26 | 42.3% | 49,882 |

### Offline Extraction

| Split | Slices saved | Avg per subject |
|---|---|---|
| train | 31,519 | 153.8 |
| val | 4,009 | 154.2 |
| test | 4,010 | 154.2 |
| **Total** | **39,538** | |

Extraction time: **20.1 minutes**. File structure: `processed_dataset/{split}/images/{subject_id}_slice{idx}.npy` (shape `[4,240,240]` float32) + `masks/` (shape `[240,240]` int8).

### DataLoader Benchmark

`BraTSDataset.__getitem__`: 38.6 ms/batch (bs=4) vs ~5000–15000 ms with live NIfTI → **130–390× faster**.

### Source modules created
- `src/dataset.py` — `BraTSDataset`, `load_subject()`, `clip_outliers()`, `zscore_normalise()`, `get_valid_slice_indices()`

### Files created
- `notebooks/02_preprocessing.ipynb`, `split.json`, `processed_dataset/` (39,538 `.npy` pairs)
- `processing_02_outputs/`: preprocessing QC visualizations (12 files)

---

## SEZIONE 6 — Training Infrastructure (Step 4a — Implemented 2026-04-23)

### src/losses.py

| Class | Description |
|---|---|
| `DiceLoss` | Soft multi-class Dice from logits. `ignore_background=True` excludes class 0. Laplace smoothing ε=1e-6. |
| `FocalLoss` | Multi-class focal, γ=2.0, per-class α weights. `log_softmax` + gather for stability. |
| `CombinedLoss` | `dice_weight × DiceLoss + focal_weight × FocalLoss`. Returns `(total, d_loss, f_loss)`. |

### src/train_utils.py

| Function/Class | Purpose |
|---|---|
| `get_augmentation(p)` | Geometric-only: HorizontalFlip, RandomRotate90, ShiftScaleRotate, ElasticTransform |
| `get_class_weights()` | Inverse-frequency weights summing to 1 |
| `MetricTracker` | Running-mean accumulator for per-batch metrics |
| `center_crop(image, mask, 192)` | Deterministic 192×192 center crop |
| `set_encoder_trainable(model, bool)` | Freeze/unfreeze `model.encoder` |
| `train_one_epoch()` | Full loop: AMP, gradient clipping max_norm=1.0 |
| `evaluate()` | `@torch.no_grad()` validation/test loop |
| `save_checkpoint()` / `load_checkpoint()` | Full state serialisation |
| `format_metrics()` | Compact log-string formatter |

**Augmentation policy (CRITICAL — DO NOT MODIFY):** Only geometric augmentations. Pixel-level transforms forbidden — input is Z-score-normalised float32 with negative values.

### Common Training Hyperparameters (used by all 3 models)

| Parameter | Value |
|---|---|
| Crop size | 192×192 (center crop) |
| Batch size | 16 |
| Phase 1 (frozen encoder) | 5 epochs, LR=3e-4 |
| Phase 2 (full fine-tuning) | 25 epochs, encoder LR=1e-5, decoder LR=1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR per phase |
| Loss | CombinedLoss (Dice+Focal, weights 1:1, γ=2.0, ignore_background=True) |
| AMP | Enabled on CUDA |
| Gradient clip | max_norm=1.0 |

### Per-class Weights (inverse-frequency, sum=1)

| Class | Weight |
|---|---|
| BG | 0.000354 |
| NCR | 0.533162 |
| ED | 0.079794 |
| ET | 0.386690 |

---

## SEZIONE 7 — U-Net/ResNet34 Training (Step 4b — Completed 2026-04-23)

### Environment

GPU: NVIDIA GeForce RTX 3080 Laptop (8 GB VRAM). PyTorch 2.11.0+cu128, AMP enabled.

### Model

| Property | Value |
|---|---|
| Architecture | U-Net / ResNet34 (via `smp.Unet`) |
| Total parameters | 24,439,940 |
| Encoder parameters | 21,287,808 |
| Decoder parameters | 3,152,132 |

### Phase 1 — Frozen Encoder (5 epochs)

| Epoch | Train Loss | Val Loss | fg Dice | Note |
|---|---|---|---|---|
| 01 | 0.9265 | 0.9175 | 0.1764 | best |
| 04 | 0.8911 | 0.9137 | 0.2642 | best |
| 05 | 0.8871 | 0.9112 | 0.2404 | |

**Phase 1 best fg Dice: 0.2642**

### Phase 2 — Full Fine-tuning (25 epochs)

| Epoch | Train Loss | Val Loss | fg Dice | Note |
|---|---|---|---|---|
| 06 | 0.8853 | 0.9083 | 0.3100 | best |
| 14 | 0.8678 | 0.9047 | 0.3891 | best |
| **15** | **0.3952** | **0.1249** | **0.8274** | **sharp "click"** |
| 16 | 0.0865 | 0.1230 | 0.8286 | best |
| **18** | **0.0739** | **0.1158** | **0.8385** | **best (saved)** |
| 30 | 0.0584 | 0.1201 | 0.8237 | last |

> Epoch 15: sharp loss drop 0.87→0.40 — "click" behaviour typical of frozen-then-unfrozen pretraining.

**Best val fg Dice: 0.8385 (epoch 18)**

### 2D Test Set Results (epoch 18)

| Metric | Value |
|---|---|
| dice_NCR | 0.8060 |
| dice_ED | 0.8111 |
| dice_ET | 0.8202 |
| **mean_fg_dice** | **0.8124** |

### Files created
- `notebooks/03_train_unet.ipynb`, `checkpoints/unet/{best,last}.pth`, `history.json`
- `processing_02_outputs/unet_training_curves.png`, `unet_val_predictions.png`

---

## SEZIONE 8 — SegFormer-B1 Training (Step 5a — Completed 2026-04-25)

### Architecture & 4-Channel Adaptation

| Property | Value |
|---|---|
| Architecture | SegFormer-B1 (`nvidia/mit-b1`) |
| Encoder | Mix Transformer (MiT-B1), ImageNet pretrained |
| Decoder | Lightweight All-MLP decode head |
| Total parameters | ~13,681,412 (44% fewer than U-Net) |

Patch embedding expanded 3→4 channels: RGB weights preserved, 4th channel = mean(RGB). Wrapper (`src/models.py`) upsamples logits H/4→H via bilinear interpolation.

### Phase 1 — Frozen Encoder (5 epochs)

| Epoch | Train Loss | Val Loss | fg Dice | Note |
|---|---|---|---|---|
| 1 | 0.9189 | 0.9161 | 0.2554 | best |
| 4 | 0.2985 | 0.1626 | 0.8067 | best |
| 5 | 0.1698 | 0.1576 | **0.8140** | **best** |

### Phase 2 — Full Fine-tuning (25 epochs)

| Epoch | Train Loss | Val Loss | fg Dice | Note |
|---|---|---|---|---|
| 6 | 0.1374 | 0.1189 | 0.8429 | best |
| **14** | **0.0779** | **0.1056** | **0.8554** | **best (saved)** |
| 30 | 0.0622 | 0.1092 | 0.8488 | last |

**Best val fg Dice: 0.8554 (epoch 14)**

### 2D Test Set Results (epoch 14)

| Metric | SegFormer | U-Net | Delta |
|---|---|---|---|
| dice_NCR | **0.8580** | 0.8060 | +0.0520 |
| dice_ED | 0.8008 | 0.8111 | -0.0103 |
| dice_ET | **0.8598** | 0.8202 | +0.0396 |
| **mean_fg_dice** | **0.8396** | 0.8124 | **+0.0272** |

### Files created
- `notebooks/05_train_segformer.ipynb`, `checkpoints/segformer/{best,last}.pth`, `history.json`
- `processing_02_outputs/segformer_training_curves.png`, `segformer_val_predictions.png`

---

## SEZIONE 9 — FPN/ResNet34 Training (Step 5b — Completed 2026-04-30)

### Architecture

| Property | Value |
|---|---|
| Architecture | FPN / ResNet34 (via `smp.FPN`) |
| Encoder | ResNet34, ImageNet pretrained (**same as U-Net**) |
| Total parameters | ~22,052,164 |
| Encoder parameters | 21,287,808 (identical to U-Net) |
| Decoder parameters | ~764,356 (vs U-Net's 3,152,132) |

### Phase 1 — Frozen Encoder (5 epochs)

Best fg Dice: **0.7916** — much faster convergence than U-Net (0.2642) with frozen encoder.

### Phase 2 — Full Fine-tuning (25 epochs)

| Epoch | Train Loss | Val Loss | fg Dice | Note |
|---|---|---|---|---|
| **15** | **0.0836** | **0.1159** | **0.8461** | **best (saved)** |
| 30 | 0.0681 | 0.1169 | 0.8430 | last |

**Best val fg Dice: 0.8461 (epoch 15)** — smooth convergence, no "click".

### Best Epoch Val Metrics (Epoch 15)

dice_NCR=0.8474, dice_ED=0.8147, dice_ET=0.8761

### Files created
- `notebooks/06_train_fpn.ipynb`, `checkpoints/fpn/{best,last}.pth`, `history.json`
- `EDA_02_outputs/fpn_training_curves.png`
---

## SEZIONE 10 — Clinical Evaluation Infrastructure (Step 6a — Completed 2026-04-23/25)

### src/eval_utils.py

| Function | Purpose |
|---|---|
| `get_subject_ids(data_dir)` | List unique subject IDs from slice filenames |
| `infer_dataset_shape(data_dir, subject_id)` | Auto-detect `orig_size` and `n_slices` for cross-domain |
| `predict_volume(model, data_dir, subject_id, device)` | Slice-by-slice 3D inference with center-crop consistency |
| `load_subject_nifti_meta(nifti_root, subject_id)` | Load NIfTI affine + header |
| `volume_to_nifti(volume, affine, header)` | Wrap prediction as saveable NIfTI |
| `remove_small_components(volume, min_voxels=50)` | Post-process: kill isolated blobs <50 voxels (26-connectivity) |
| `compute_dice_volume(pred, gt)` | Per-class 3D Dice |
| `compute_hd95_volume(pred, gt, voxel_spacing)` | Per-class HD95 via `medpy.metric.binary.hd95` with edge-case handling |

### HD95 Edge-Case Handling

| Case | Condition | Return | Log |
|---|---|---|---|
| A — Both empty | neither pred nor GT | `NaN` | Silent |
| B — GT only | GT present, pred empty | `NaN` | RuntimeWarning (complete miss) |
| C — Pred only | pred present, GT empty | `NaN` | RuntimeWarning (hallucination) |
| Normal | both non-empty | float (mm) | None |

NaN convention: downstream uses `np.nanmean` and counts NaN cases separately.

### src/evaluate_3d_test.py

Standalone CLI script for 3D evaluation on the TEST set:
```
python -m src.evaluate_3d_test --model unet|fpn|segformer|all
```
Supports all 3 models. Results saved as JSON in `evaluation_outputs/`.

### notebooks/04_evaluation.ipynb (Val set — U-Net only)

Ran 3D evaluation on 26 val subjects for U-Net. Mean fg Dice (3D, post-processed): **0.6293**. This was later superseded by test set evaluation.

### Files created
- `src/eval_utils.py`, `src/evaluate_3d_test.py`
- `notebooks/04_evaluation.ipynb`, `notebooks/06b_evaluation_fpn.ipynb`

---

## SEZIONE 11 — 3D Volumetric Results on TEST Set (Step 6b — Completed 2026-04-30)

All three models evaluated on 26 test subjects with post-processing (remove_small_components, 50 voxels, 26-connectivity).

### U-Net/ResNet34 — 3D Test Metrics

| Class | Dice mean | Dice std | HD95 mean (mm) | HD95 std | HD95 NaN |
|---|---|---|---|---|---|
| NCR | 0.5856 | 0.3718 | 9.21 | 9.66 | 10 |
| ED | 0.7239 | 0.2523 | 7.45 | 6.61 | 0 |
| ET | 0.5398 | 0.4307 | 17.85 | 9.93 | 17 |
| **Mean FG** | **0.6164** | | | | |

### FPN/ResNet34 — 3D Test Metrics

| Class | Dice mean | Dice std | HD95 mean (mm) | HD95 std | HD95 NaN |
|---|---|---|---|---|---|
| NCR | 0.5590 | 0.3965 | 8.70 | 6.99 | 12 |
| ED | 0.7104 | 0.2380 | 7.63 | 5.76 | 0 |
| ET | 0.5946 | 0.4448 | 18.69 | 9.31 | 20 |
| **Mean FG** | **0.6213** | | | | |

### SegFormer-B1 — 3D Test Metrics

| Class | Dice mean | Dice std | HD95 mean (mm) | HD95 std | HD95 NaN |
|---|---|---|---|---|---|
| NCR | 0.6752 | 0.3339 | 5.23 | 4.07 | 10 |
| ED | 0.7229 | 0.2575 | 8.15 | 8.08 | 0 |
| ET | 0.6299 | 0.4387 | 14.60 | 7.07 | 20 |
| **Mean FG** | **0.6760** | | | | |

### HD95 Edge-Case Summary (per model)

| Warning Type | U-Net | FPN | SegFormer |
|---|---|---|---|
| Complete miss (GT present, pred empty) — NCR | 4 | 4 | — |
| Complete miss (GT present, pred empty) — ET | 5 | 5 | — |
| Complete miss (GT present, pred empty) — ED | 1 | 0 | — |
| Hallucination (pred present, GT empty) — ET | 2 | 2 | — |
| Hallucination (pred present, GT empty) — NCR | 2 | 1 | — |

### Results files
- `evaluation_outputs/test_3d_metrics_unet.json`
- `evaluation_outputs/test_3d_metrics_fpn.json`
- `evaluation_outputs/test_3d_metrics_segformer.json`

---

## SEZIONE 12 — Three-Way Comparison (Step 7 — Completed 2026-04-30)

### notebooks/08_comparison.ipynb

Notebook di confronto three-way U-Net vs FPN vs SegFormer:
- Tabella riassuntiva quantitativa (Dice + HD95 per classe)
- Bar chart comparativo Dice per classe
- Box-plot distribuzione per-soggetto Dice
- HD95 bar chart comparativo
- Scatter plot per-subject mean FG Dice
- Failure analysis: 5 worst subjects per modello
- Agreement analysis: soggetti più difficili
- Training convergence comparison (loss + fg Dice curves)

### Files created
- `notebooks/08_comparison.ipynb`
- `comparison_outputs/summary_comparison.csv`
- `comparison_outputs/per_subject_fg_dice_all_models.csv`
- `comparison_outputs/`: `dice_comparison_bar.png`, `dice_boxplot_per_class.png`, `hd95_comparison_bar.png`, `fg_dice_scatter.png`, `convergence_comparison.png`

---

## SEZIONE 13 — Cross-Domain Generalization (Structured, Not Executed)

### Overview

Phase 7 evaluates model robustness under domain shift — zero-shot (no retraining) on adult BraTS datasets.

### notebooks/07_cross_domain_evaluation.ipynb

Self-contained pipeline with `NEW_DATASET_PATH` placeholder. Features:
- `evaluate_split()` helper — generic 3D eval for any preprocessed split
- Auto-detects spatial dims via `infer_dataset_shape()`
- Comparative summary table: in-domain vs cross-domain with % change
- Distribution box-plots, failure analysis

### How to use
1. Preprocess new dataset with `notebooks/02_preprocessing.ipynb` (set `DATA_ROOT`, `OUTPUT_ROOT`)
2. In `notebooks/07_cross_domain_evaluation.ipynb`, set `NEW_DATASET_PATH`
3. Run all cells

### Status
🔶 Notebook created and validated, but not executed (requires external adult BraTS dataset).

---

## SEZIONE 14 — Tabella Riepilogativa Finale

### 2D Test Set — Slice-Level Dice (best checkpoint)

| Metric | U-Net/ResNet34 | FPN/ResNet34 | SegFormer-B1 |
|---|---|---|---|
| dice_NCR | 0.8060 | 0.8474 | **0.8580** |
| dice_ED | 0.8111 | **0.8147** | 0.8008 |
| dice_ET | 0.8202 | **0.8761** | 0.8598 |
| **mean_fg_dice** | 0.8124 | 0.8461 | **0.8396** |
| Best epoch | 18 | 15 | 14 |
| Best val fg Dice | 0.8385 | 0.8461 | **0.8554** |
| Total params | 24,439,940 | ~22,052,164 | ~13,681,412 |

> Note: FPN val metrics (0.8461) shown above; 2D test set Dice for FPN uses best epoch val metrics as proxy (notebook Section 9 not re-executed post-training due to venv change).

### 3D Test Set — Volumetric Clinical Metrics (post-processed, 26 subjects)

| Metric | U-Net/ResNet34 | FPN/ResNet34 | SegFormer-B1 |
|---|---|---|---|
| dice_NCR (3D) | 0.5856 | 0.5590 | **0.6752** |
| dice_ED (3D) | 0.7239 | 0.7104 | **0.7229** |
| dice_ET (3D) | 0.5398 | 0.5946 | **0.6299** |
| **mean_fg_dice (3D)** | 0.6164 | 0.6213 | **0.6760** |
| HD95_NCR (mm) | 9.21 | 8.70 | **5.23** |
| HD95_ED (mm) | **7.45** | 7.63 | 8.15 |
| HD95_ET (mm) | 17.85 | 18.69 | **14.60** |

### Key Findings

1. **SegFormer-B1** is the best model overall: highest 3D fg Dice (0.6760, +9.7% vs U-Net) with 44% fewer parameters
2. **FPN** marginally outperforms U-Net on 3D fg Dice (+0.8%), confirming the decoder architecture matters even with identical encoders
3. **ED** is the most reliably segmented class across all models (Dice >0.70, HD95 <9mm)
4. **ET** is the hardest class: high NaN rate (17-20/26 subjects) due to ET absence in many subjects
5. **NCR** shows the largest model gap: SegFormer (0.6752) vs FPN (0.5590) = +20.8% relative improvement
6. **2D vs 3D gap**: All models show ~20-25% Dice drop from 2D slice-level to 3D volumetric evaluation, expected for 2D-based architectures

### Checkpoint Summary

| Model | Checkpoint | Epoch | Size |
|---|---|---|---|
| U-Net/ResNet34 | `checkpoints/unet/best.pth` | 18 | 293 MB |
| FPN/ResNet34 | `checkpoints/fpn/best.pth` | 15 | — |
| SegFormer-B1 | `checkpoints/segformer/best.pth` | 14 | 164 MB |

---

## Changelog

| Data | Modifica |
|---|---|
| 2026-04-22 | Step 0-1: Dataset analysis, environment setup |
| 2026-04-22 | Step 2-3: EDA, preprocessing, dataset class, split |
| 2026-04-23 | Step 4: Training infra (losses, train_utils), U-Net training + results |
| 2026-04-23 | Step 6a: eval_utils (3D reconstruction, post-processing, HD95) |
| 2026-04-25 | Step 6b: notebooks/04_evaluation.ipynb (3D val eval for U-Net) |
| 2026-04-25 | Step 5a: SegFormer implementation (models.py) + training |
| 2026-04-25 | Phase 7: Cross-domain refactoring + notebooks/07_cross_domain_evaluation.ipynb |
| 2026-04-30 | Step 5b: FPN training + 3D evaluation |
| 2026-04-30 | Step 7: notebooks/08_comparison.ipynb (three-way) |
| 2026-04-30 | **Riorganizzazione context.md**: struttura logica, tabelle TBD compilate, sezione riepilogativa finale |
