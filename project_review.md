# 🧠 Project Review — BraTS-PEDs Brain Tumour Segmentation

**Reviewer**: Tech Lead / Senior Software Engineer  
**Date**: 2026-04-29  
**Scope**: Full codebase audit vs `CONTEXT.md` specifications

> **Status (2026-05-11):** historical document. Every 🔴 / 🟡 item flagged below has been closed. For the up-to-date state of the project, see `current_progress_review.md` and `next_steps_action_plan.md`.

---

## 1. Stato Attuale del Progetto — Panoramica

| Fase (da CONTEXT.md) | Step | Stato | Note |
|---|---|---|---|
| **Fase 1** — Data Acquisition & NIfTI Parsing | Step 0–2 | ✅ Completata | EDA notebook eseguito con successo |
| **Fase 2** — 3D→2D Conversion & Preprocessing | Step 3 | ✅ Completata | 39.538 slice `.npy` estratte, split stratificato |
| **Fase 3** — Baseline U-Net Training | Step 4 | ✅ Completata | U-Net/ResNet34, test fg Dice = 0.8124 |
| **Fase 4** — Secondo Modello (FPN → SegFormer) | Step 5 | ⚠️ **Deviazione** | Implementato **SegFormer-B1** invece di **FPN** |
| **Fase 5** — Clinical Evaluation & Visualization | Step 6 | ✅ Parzialmente completata | Evaluation 3D solo su **val set** (non test set) |
| **Fase 5** — Comparison & Report | Step 7 | ❌ Non implementata | Nessun report comparativo finale |
| **Extra** — Cross-Domain Generalization | Phase 7 | 🔶 Strutturato, non eseguito | Notebook creato ma placeholder `NEW_DATASET_PATH` |

---

## 2. Moduli Implementati con Successo

### 2.1 Moduli Sorgente (`src/`)

| File | LOC | Qualità | Commento |
|---|---|---|---|
| [dataset.py](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/src/dataset.py) | 247 | ⭐⭐⭐⭐⭐ | Eccellente. Docstrings complete, type hints, edge-case handling per volumi degeneri. Pipeline di normalizzazione (clip P99.5 → Z-score → background=0) ben ragionata e documentata. |
| [losses.py](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/src/losses.py) | 304 | ⭐⭐⭐⭐⭐ | `DiceLoss`, `FocalLoss`, `CombinedLoss` implementati correttamente. Numerically stable (log_softmax + gather). Return tuple `(total, d_loss, f_loss)` per logging indipendente. |
| [train_utils.py](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/src/train_utils.py) | 479 | ⭐⭐⭐⭐ | Solido. AMP support, gradient clipping, MetricTracker ben progettato. Augmentation policy correttamente limitata a trasformazioni geometriche. |
| [eval_utils.py](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/src/eval_utils.py) | 568 | ⭐⭐⭐⭐⭐ | Modulo più sofisticato. HD95 con edge-case handling esemplare (Cases A/B/C). Post-processing con 26-connectivity. Cross-domain support tramite `infer_dataset_shape()`. |
| [models.py](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/src/models.py) | 162 | ⭐⭐⭐⭐ | 4-channel adaptation elegante (mean-of-RGB per il 4° canale). SegFormerWrapper assicura drop-in compatibility con il training loop. |

### 2.2 Notebooks

| Notebook | Stato | Output Confermati |
|---|---|---|
| [01_EDA.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/01_EDA.ipynb) | ✅ Eseguito | 4 plot in `EDA_01_outputs/` |
| [02_preprocessing.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/02_preprocessing.ipynb) | ✅ Eseguito | `processed_dataset/` (39.538 slice), `split.json` |
| [03_train_unet.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/03_train_unet.ipynb) | ✅ Eseguito | `checkpoints/unet/best.pth`, training curves |
| [04_evaluation.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/04_evaluation.ipynb) | ✅ Eseguito | 3D Dice + HD95 su **val set**, failure analysis |
| [05_train_segformer.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/05_train_segformer.ipynb) | ✅ Eseguito | `checkpoints/segformer/best.pth`, training curves |
| [07_cross_domain_evaluation.ipynb](file:///c:/Users/lrnzp/OneDrive/Desktop/POLITO/MAGISTRALE/Secondo%20Anno/Primo%20Semestre/AI%20in%20Medicine/Project/notebooks/07_cross_domain_evaluation.ipynb) | 🔶 Creato, non eseguito | Placeholder, nessun output |

### 2.3 Artefatti e Checkpoint

- **Checkpoint U-Net**: `checkpoints/unet/best.pth` (293 MB, epoch 18) + `last.pth` + `history.json` ✅
- **Checkpoint SegFormer**: `checkpoints/segformer/best.pth` (164 MB, epoch 14) + `last.pth` + `history.json` ✅
- **Split data**: `split.json` — 205 train / 26 val / 26 test (Strategy C, SEED=42) ✅
- **Visualizzazioni**: 12 file PNG in `processing_02_outputs/`, 4 file in `EDA_01_outputs/` ✅

---

## 3. Discrepanze tra Codebase e CONTEXT.md

### 3.1 🔴 Discrepanza Critica — FPN sostituita da SegFormer

> [!WARNING]
> Il `CONTEXT.md` Sezione 2 (Roadmap) e Sezione 4 (Implementation Plan, Step 5) specificano **FPN (Feature Pyramid Network)** come secondo modello con lo stesso backbone ResNet34 per un confronto equo. Invece è stato implementato **SegFormer-B1** con MiT-B1 encoder.

**Impatto**:
- Il confronto **non è a parità di encoder**: ResNet34 (21.3M params) vs MiT-B1 (13.7M totali) → non è un confronto "head-only"
- Il `CONTEXT.md` Sezione 1 specifica esplicitamente: *"Verranno confrontate la 2D U-Net e la Feature Pyramid Network (FPN)"*
- La Fase 4 richiede *"mantenendo lo stesso encoder ResNet34 per equità di confronto"*

**Valutazione**: SegFormer è un modello più avanzato e i risultati sono migliori (+2.7% fg Dice), ma l'aderenza alle specifiche di progetto è parzialmente violata. Il notebook `04_train_fpn.ipynb` (menzionato come Step 5 in CONTEXT.md) **non esiste**.

### 3.2 🔴 Valutazione 3D eseguita su val set, non test set

> [!WARNING]
> La `04_evaluation.ipynb` (Sezione 10-11 di CONTEXT.md) esegue la valutazione clinica 3D su **26 soggetti val**, non sui **26 soggetti test**. Il CONTEXT.md Step 6 specifica chiaramente: *"Calcolo inferenziale sul Test set"*. 

**Impatto**:
- I risultati 3D riportati (mean fg Dice = 0.6293) sono sul **validation set**, non il test set
- Le metriche test riportate nella Sezione 9 (0.8124 fg Dice) sono **2D slice-level**, non 3D volumetriche
- Manca una valutazione 3D **definitiva** sul test set per entrambi i modelli

### 3.3 🟡 Notebook mancante — `06_comparison.ipynb`

Il CONTEXT.md Step 7 prevede:
- Tabella quantitativa comparativa U-Net vs secondo modello
- Overlay qualitativi che evidenzino errori ai bordi del tumore

**Stato**: Non esiste nessun notebook `06_*.ipynb` di confronto. La comparazione SegFormer vs U-Net è limitata a una tabella inline dentro `05_train_segformer.ipynb` (sezione 9), basata su metriche **2D test**.

### 3.4 🟡 `requirements.txt` incompleto

Il file `requirements.txt` **non include** `transformers` e `accelerate`, necessari per SegFormer (`src/models.py`):
```diff
 # Deep Learning
 torch
 torchvision
+transformers
+accelerate
```

### 3.5 🟡 Output directory inconsistency

Il CONTEXT.md e i notebook usano `EDA_02_outputs/` come directory di output, ma la directory effettiva su disco si chiama `processing_02_outputs/`. Tutti i 12 file di output sono in `processing_02_outputs/`, non `EDA_02_outputs/`.

### 3.6 🟢 Discrepanza minore — Split subjects count

Il CONTEXT.md Step 3 specifica "206 train / 26 val / 25 test", ma l'effettivo split implementato è **205 train / 26 val / 26 test** (confermato da `split.json`). La differenza di 1 soggetto tra train e test è insignificante e ben documentata nella Sezione 7.

### 3.7 🟢 Virtual environments proliferation

Ci sono **5 directory `.venv`** nella root del progetto (`.venv`, `.venv-1`, `.venv-2`, `.venv-3`, `.venv-4`). Questo suggerisce problemi ricorrenti con l'ambiente Python, probabilmente legati a incompatibilità tra Python 3.14 e alcune dipendenze.

---

## 4. Qualità Generale del Codice

### Punti di Forza

1. **Docstrings esemplari**: Ogni funzione pubblica ha docstring con `Args`, `Returns`, `Raises`, rationale clinico e decisioni di design. Questo è raro e lodevole.
2. **Type hints completi**: Tutti i moduli usano `typing` annotations, incluse generics e `Optional`.
3. **Edge-case handling robusto**: In particolare `compute_hd95_volume()` gestisce 4 scenari distinti (Cases A/B/C + eccezioni impreviste) — design professionale.
4. **Separazione delle responsabilità**: `dataset.py` (I/O + preprocessing), `losses.py` (funzioni di costo), `train_utils.py` (training loop), `eval_utils.py` (inference 3D + metriche cliniche), `models.py` (architetture). Architettura pulita.
5. **Augmentation policy**: La restrizione a sole trasformazioni geometriche è ben motivata e documentata con warning "CRITICAL — DO NOT MODIFY".
6. **Riproducibilità**: `SEED=42` applicato consistentemente; split strategy documentata e salvata in `split.json`.

### Punti Deboli

1. **`CONTEXT.md` monolitico** (53 KB, 1076 righe): Funziona sia da specifica che da logbook. È diventato molto lungo e difficile da navigare. Sarebbe meglio separare specifica e log di esecuzione.
2. **Nessun test unitario**: Non esiste una directory `tests/` con test per le funzioni critiche (`DiceLoss`, `FocalLoss`, `remove_small_components`, `predict_volume`).
3. **Warning di import ripetuti**: In `eval_utils.py`, `import warnings` è ripetuto 3 volte dentro il corpo di `compute_hd95_volume()` (righe 524, 538, 558) invece di essere importato una volta in testa al file.
4. **Duplicazione di costanti**: `NUM_CLASSES=4`, `CLASS_NAMES` sono definite sia in `train_utils.py` (riga 47-48) che in `eval_utils.py` (riga 59-60). Dovrebbero provenire da un singolo punto (`dataset.py`).
5. **No `.gitignore`**: Manca un `.gitignore` per escludere `__pycache__/`, `.venv*/`, `processed_dataset/`, `checkpoints/`, file `.npy` pesanti.

---

## 5. Action Plan — Prossimi Step

### 🔴 Priorità Alta — Azioni Immediate

#### 5.1 Implementare la valutazione 3D sul TEST set

La valutazione 3D è stata eseguita solo sul validation set. È necessario rieseguire la stessa pipeline sui 26 soggetti test **per entrambi i modelli** (U-Net e SegFormer).

- [x] **Creato `src/evaluate_3d_test.py`** — script standalone che punta a `processed_dataset/test` e supporta entrambi i modelli (U-Net e SegFormer). Salva risultati JSON in `evaluation_outputs/`. *(alternativa più modulare alla duplicazione del notebook)*
- [x] Calcolare Dice 3D + HD95 per classe su test set, per U-Net (best.pth, epoch 18) — **FATTO**: mean fg Dice = **0.6164**, risultati salvati in `evaluation_outputs/test_3d_metrics_unet.json`
- [x] Calcolare Dice 3D + HD95 per classe su test set, per SegFormer (best.pth, epoch 14) — **FATTO**: mean fg Dice = **0.6760**, risultati salvati in `evaluation_outputs/test_3d_metrics_segformer.json`
- [x] Eseguire post-processing (`remove_small_components`) + failure analysis su test set — **FATTO**: post-processing integrato in `evaluate_3d_test.py`; failure analysis visuale sarà nel notebook `06_comparison.ipynb`

#### 5.2 Implementare FPN ✅ FATTO (Opzione A — 2026-04-30)

**Scelta: Opzione A** — Implementato FPN come da specifica + mantenuto SegFormer come confronto aggiuntivo.

- [x] Creato `notebooks/06_train_fpn.ipynb` con `smp.FPN(encoder_name='resnet34', ...)`
- [x] Addestrato con gli stessi iperparametri di U-Net (30 epochs, two-phase)
- [x] Best val fg Dice: **0.8461** (epoch 15) — supera U-Net (0.8385)
- [x] 3D test fg Dice: **0.6213** — supera U-Net (0.6164), sotto SegFormer (0.6760)
- [x] Aggiornato `src/evaluate_3d_test.py` con supporto FPN
- [x] Creato `notebooks/06b_evaluation_fpn.ipynb` per valutazione 3D
- [x] Risultati salvati in `evaluation_outputs/test_3d_metrics_fpn.json`

#### 5.3 Aggiornare `requirements.txt` ✅ FATTO

```diff
 # Segmentation models with pretrained backbones
 segmentation-models-pytorch

+# Transformers (SegFormer)
+transformers
+accelerate
```

---

### 🟡 Priorità Media — Sviluppi Successivi

#### 5.4 Creare il notebook di confronto finale ✅ FATTO (2026-04-30)

Creato `notebooks/08_comparison.ipynb` — confronto three-way U-Net vs FPN vs SegFormer:

- [x] Tabella riassuntiva quantitativa: Dice e HD95 per classe per tutti e 3 i modelli
- [x] Bar chart comparativo Dice per classe
- [x] Box-plot distribuzione per-soggetto Dice per classe
- [x] HD95 bar chart comparativo
- [x] Scatter plot per-subject mean FG Dice
- [x] Failure analysis: 5 worst subjects per modello
- [x] Agreement analysis: soggetti piu difficili per tutti i modelli
- [x] Training convergence comparison (loss + fg Dice curves)
- [x] Salvare CSV e figure in `comparison_outputs/`

#### 5.5 Valutazione cross-domain (se si dispone di un dataset adulto BraTS)

- [ ] Procurarsi un dataset BraTS adulto (es. BraTS2023-GLI)
- [ ] Preprocessarlo con la stessa pipeline (`notebooks/02_preprocessing.ipynb`)
- [ ] Eseguire `notebooks/07_cross_domain_evaluation.ipynb` impostando `NEW_DATASET_PATH`
- [ ] Documentare il domain shift nelle performance

#### 5.6 Ricostruzione 3D e visualizzazione NIfTI

Il CONTEXT.md Step 6 prevede: *"Ricostruzione delle predizioni 2D in volumi 3D per generare visualizzazioni cliniche qualitative"*.

- [ ] Salvare predizioni 3D come file `.nii.gz` con affine corretto (le utility `volume_to_nifti()` e `load_subject_nifti_meta()` esistono già)
- [ ] Includere nel report screenshot da un viewer NIfTI (es. ITK-SNAP, 3D Slicer)

---

### 🟢 Suggerimenti / Miglioramenti Opzionali

#### 5.7 Refactoring — Centralizzare costanti

```python
# src/constants.py (NEW)
NUM_CLASSES = 4
CLASS_NAMES = ("background", "NCR", "ED", "ET")
CROP_SIZE = 192
ORIG_SIZE = 240
N_SLICES = 155
```

Poi importare da `src.constants` in `train_utils.py`, `eval_utils.py`, `models.py`.

#### 5.8 Aggiungere `.gitignore`

```gitignore
__pycache__/
*.pyc
.venv*/
processed_dataset/
checkpoints/
*.npy
*.pth
.ipynb_checkpoints/
```

#### 5.9 Pulizia virtual environments

Rimuovere le 4 directory `.venv-1` ... `.venv-4` non utilizzate, mantenere solo `.venv`.

#### 5.10 Test unitari per funzioni critiche

Creare `tests/` con almeno:
- `test_losses.py` — verificare `DiceLoss`, `FocalLoss` su tensori noti
- `test_eval_utils.py` — verificare `compute_dice_volume` e `remove_small_components` su volumi sintetici
- `test_dataset.py` — verificare `clip_outliers`, `zscore_normalise`

#### 5.11 Fix minor: `import warnings` in `eval_utils.py`

Spostare `import warnings` dal corpo di `compute_hd95_volume()` (righe 524, 538, 558) all'inizio del file, insieme agli altri import.

#### 5.12 IoU mancante nelle metriche

Il CONTEXT.md Step 6 menziona *"Metriche: Dice per class, mean IoU, HD95"*. **IoU non è calcolato** da nessuna parte. Aggiungere `compute_iou_volume()` in `eval_utils.py`:

```python
def compute_iou_volume(pred, gt, smooth=1e-6):
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = float((pred_c & gt_c).sum())
        union = float((pred_c | gt_c).sum())
        iou = (inter + smooth) / (union + smooth)
        results[f"iou_{name}"] = iou
    return results
```

---

## 6. Riepilogo delle Metriche Attuali

### Test Set (2D slice-level)

| Modello | dice_NCR | dice_ED | dice_ET | **mean fg Dice** |
|---|---|---|---|---|
| U-Net/ResNet34 | 0.8060 | 0.8111 | 0.8202 | **0.8124** |
| SegFormer-B1 | 0.8580 | 0.8008 | 0.8598 | **0.8396** (+2.7%) |

### Val Set (3D volumetric, post-processed) — Solo U-Net

| Classe | Dice mean | HD95 mean (mm) | HD95 NaN |
|---|---|---|---|
| NCR | 0.5917 | 10.70 | 12/26 |
| ED | 0.7003 | 7.22 | 1/26 |
| ET | 0.5959 | 16.19 | 19/26 |
| **Mean fg** | **0.6293** | — | — |

> [!CAUTION]
> Mancano completamente le metriche 3D sul **test set** e le metriche 3D per **SegFormer**. Queste sono essenziali per il report finale.

---

## 7. Valutazione Complessiva

| Aspetto | Voto | Commento |
|---|---|---|
| Architettura codice | ⭐⭐⭐⭐⭐ | Modulare, pulita, ben separata |
| Documentazione | ⭐⭐⭐⭐⭐ | Docstrings eccezionali, CONTEXT.md dettagliato |
| Aderenza alle specifiche | ⭐⭐⭐ | FPN mancante, eval su val invece di test |
| Completezza pipeline | ⭐⭐⭐⭐ | Training + eval funzionanti, manca report finale |
| Risultati modelli | ⭐⭐⭐⭐ | 0.84 fg Dice (2D) è un risultato competitivo |
| Riproducibilità | ⭐⭐⭐⭐ | Seed fissato, split salvato, ma requirements incompleto |

**Verdetto**: Il progetto è in uno stato **avanzato** (~75% completato). Il codice è di alta qualità e le fondamenta sono solide. Le lacune principali sono: (1) mancanza della valutazione 3D sul test set per entrambi i modelli, (2) assenza di FPN o giustificazione formale della sostituzione, (3) assenza del notebook di confronto finale. Queste sono tutte completabili in 1-2 sessioni di lavoro.
