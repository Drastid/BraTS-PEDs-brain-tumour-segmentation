# BraTS-PEDs Brain Tumour Segmentation

![Tumour segmentation overlay](img/TumorSeg.png)

Automated semantic segmentation of paediatric brain gliomas from multi-parametric MRI, comparing three deep learning architectures on the [BraTS-PEDs-v1](https://www.synapse.org/Synapse:syn51514105) dataset.

**Course**: AI in Medicine — Politecnico di Torino, MSc (A.Y. 2025-2026)

---

## Clinical Context

This project targets the delineation of the **Gross Tumor Volume (GTV)** in paediatric gliomas, a task performed daily by neuroncologists and neurosurgeons. Manual segmentation is time-consuming and suffers from inter-operator variability. Automating it with reproducible, objective predictions reduces both.

The clinical metric that matters most is **HD95** (95th-percentile Hausdorff Distance): a contour error at the tumour boundary can mean removing healthy tissue or leaving active tumour behind.

---

## Dataset

**BraTS-PEDs-v1** — 257 training subjects (NIfTI, 1mm isotropic, 240×240×155 voxels).

Each subject has four co-registered MRI sequences and a manually annotated segmentation mask:

| Modality | Description |
|---|---|
| T1c | T1 post-contrast |
| T1n | T1 native |
| T2f | T2 FLAIR |
| T2w | T2 weighted |

**Segmentation labels**: `0` Background · `1` NCR (Necrotic Core) · `2` ED/SNFH (Edema) · `3` ET (Enhancing Tumour)

**Class imbalance**: BG 99.40% · NCR 0.066% · ED 0.441% · ET 0.091%

**Split** (Strategy C — ET+Quartile, SEED=42): 205 train / 26 val / 26 test

---

## Models

All three models receive a `[B, 4, H, W]` tensor and produce `[B, 4, H, W]` logits (4 classes). Training is identical across models (same loss, same hyperparameters, same two-phase schedule).

| Model | Encoder | Decoder | Params |
|---|---|---|---|
| **U-Net** | ResNet34 (ImageNet) | Standard U-Net decoder | 24.4M |
| **FPN** | ResNet34 (ImageNet) | Feature Pyramid Network | 22.1M |
| **SegFormer-B1** | MiT-B1 (ImageNet) | Lightweight All-MLP head | 13.7M |

SegFormer required a **4-channel patch embedding adaptation**: RGB weights are preserved; the 4th channel is initialised as `mean(RGB)`. A thin wrapper (`src/models.py`) upsamples the native H/4×W/4 logits back to input resolution.

---

## Results

### 3D Volumetric Metrics — Test Set (26 subjects, post-processed)

| Metric | U-Net / ResNet34 | FPN / ResNet34 | SegFormer-B1 |
|---|---|---|---|
| Dice NCR | 0.5856 | 0.5590 | **0.6752** |
| Dice ED | 0.7239 | 0.7104 | **0.7229** |
| Dice ET | 0.5398 | 0.5946 | **0.6299** |
| **Mean FG Dice** | 0.6164 | 0.6213 | **0.6760** |
| HD95 NCR (mm) | 9.21 | 8.70 | **5.23** |
| HD95 ED (mm) | **7.45** | 7.63 | 8.15 |
| HD95 ET (mm) | 17.85 | 18.69 | **14.60** |

**SegFormer-B1** is the best model overall: +9.7% mean FG Dice vs U-Net, +20.8% relative improvement on NCR, 44% fewer parameters.

### 2D Slice-Level Metrics — Test Set (best checkpoint)

| Model | Dice NCR | Dice ED | Dice ET | Mean FG Dice | Best epoch |
|---|---|---|---|---|---|
| U-Net / ResNet34 | 0.8060 | 0.8111 | 0.8202 | 0.8124 | 18 |
| FPN / ResNet34 | 0.8474 | 0.8147 | 0.8761 | 0.8461 | 15 |
| SegFormer-B1 | **0.8580** | 0.8008 | **0.8598** | **0.8554** | 14 |

The ~20–25% Dice drop from 2D to 3D is expected for slice-based architectures with no 3D context.

---

## Training

### Loss Function

```
CombinedLoss = DiceLoss(ignore_background=True) + FocalLoss(γ=2.0)
```

Both terms weighted 1:1. Per-class inverse-frequency weights counteract the severe class imbalance: NCR weight = 0.533, ET weight = 0.387, BG weight = 0.0004.

### Two-Phase Schedule

| Phase | Epochs | Encoder | LR (encoder) | LR (decoder) |
|---|---|---|---|---|
| 1 — frozen encoder | 5 | ❄️ frozen | — | 3×10⁻⁴ |
| 2 — full fine-tuning | 25 | 🔥 trainable | 1×10⁻⁵ | 1×10⁻⁴ |

Both phases use CosineAnnealingLR, AdamW (weight decay 1×10⁻⁴), AMP, and gradient clipping (max_norm=1.0).

### Augmentation Policy

**Geometric transforms only** — pixel-level transforms (brightness, contrast) are forbidden because inputs are Z-score-normalised float32 with negative values.

```python
HorizontalFlip · RandomRotate90 · ShiftScaleRotate · ElasticTransform
```

---

## Setup

```bash
# Clone
git clone https://github.com/Drastid/BraTS-PEDs-brain-tumour-segmentation.git
cd BraTS-PEDs-brain-tumour-segmentation

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python ≥ 3.10, PyTorch ≥ 2.0 with CUDA. Tested on Python 3.14.3 + PyTorch 2.11.0+cu128 (NVIDIA RTX 3080 Laptop 8GB).

### Data Preparation

The raw NIfTI dataset and the preprocessed `.npy` slices are **not included** in this repository (too large). To reproduce:

1. Download **BraTS-PEDs-v1** from the [BraTS challenge portal](https://www.synapse.org/Synapse:syn51514105) and place it in `PKG - BraTS-PEDs-v1/`
2. Run `02_preprocessing.ipynb` — this extracts 39,538 axial slice pairs into `processed_dataset/` (~20 minutes)

---

## Usage

### Training

Run notebooks in order:

```
01_EDA.ipynb              → Exploratory data analysis
02_preprocessing.ipynb    → 3D→2D extraction + train/val/test split
03_train_unet.ipynb       → U-Net/ResNet34
05_train_segformer.ipynb  → SegFormer-B1
06_train_fpn.ipynb        → FPN/ResNet34
```

### 3D Evaluation on Test Set

```bash
# All three models
python -m src.evaluate_3d_test --model all

# Single model
python -m src.evaluate_3d_test --model unet
python -m src.evaluate_3d_test --model fpn
python -m src.evaluate_3d_test --model segformer
```

Results are saved as JSON in `evaluation_outputs/`.

### Comparison

Run `08_comparison.ipynb` for the three-way quantitative and visual comparison (Dice tables, HD95 bar charts, box-plots, failure analysis, convergence curves).

---

## Project Structure

```
.
├── src/
│   ├── dataset.py              # BraTSDataset + preprocessing utilities
│   ├── losses.py               # DiceLoss, FocalLoss, CombinedLoss
│   ├── train_utils.py          # Training loop, augmentation, checkpointing
│   ├── eval_utils.py           # 3D reconstruction, HD95, post-processing
│   ├── models.py               # SegFormerWrapper (4-channel adaptation)
│   └── evaluate_3d_test.py     # CLI evaluation script
├── 01_EDA.ipynb
├── 02_preprocessing.ipynb
├── 03_train_unet.ipynb
├── 04_evaluation.ipynb
├── 05_train_segformer.ipynb
├── 06_train_fpn.ipynb
├── 06b_evaluation_fpn.ipynb
├── 07_cross_domain_evaluation.ipynb   # Template — requires adult BraTS data
├── 08_comparison.ipynb
├── split.json                  # Reproducible subject assignment (SEED=42)
├── requirements.txt
├── EDA_01_outputs/             # EDA plots
├── EDA_02_outputs/             # FPN training curves
├── processing_02_outputs/      # Preprocessing QC + training curves
├── evaluation_outputs/         # 3D test metrics (JSON)
└── comparison_outputs/         # Comparison plots and CSVs
```

Excluded from git: `processed_dataset/` (39K `.npy` slices), `checkpoints/` (`.pth` weights), `PKG - BraTS-PEDs-v1/` (raw NIfTI).

---

## Key Implementation Details

- **Preprocessing**: clip non-zero voxels at P99.5 → Z-score normalise on brain voxels only (background forced back to 0) → label remap `4→3` (old BraTS ET convention)
- **Center crop**: 192×192 from 240×240 (`top=24, left=24`); applied identically at training time and during 3D inference (with un-crop before volume assembly)
- **Post-processing**: `remove_small_components` — removes isolated blobs < 50 voxels per class (26-connectivity) to eliminate the "confetti effect" common in 2D-based models
- **HD95 edge cases**: returns `NaN` when a class is absent in prediction or GT; downstream aggregation uses `np.nanmean`; NaN counts are reported separately
