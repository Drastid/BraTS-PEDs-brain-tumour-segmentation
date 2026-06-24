# Fase 3 — Loss con Schedulazione Dinamica (Dice-Focal + GSL)

**Data:** 2026-06-24
**Paper di riferimento:** Celaya, Riviere, Fuentes — *A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation*, arXiv:2302.03868v3.
**Obiettivo:** implementare la loss
$$\mathcal{L}(t) = \alpha(t)\,\mathcal{L}_{DiceFocal} + (1 - \alpha(t))\,\mathcal{L}_{GSL}$$
con schedulazione dinamica di $\alpha$, step scheduler, e pesi globali $w_k$ pre-calcolati offline.

---

## 1. Cosa è stato implementato

| Componente | File · simbolo | Eq. paper |
|---|---|---|
| Pesi globali $w_k$ (helper) | [src/losses.py:346](src/losses.py#L346) — `compute_global_class_weights` | Eq. 13 |
| Scheduler di $\alpha$ (linear/step/cosine) | [src/losses.py:381](src/losses.py#L381) — `AlphaScheduler` | Eq. 14–16 |
| Generalized Surface Loss | [src/losses.py:456](src/losses.py#L456) — `GeneralizedSurfaceLoss` | Eq. 12 |
| Loss combinata schedulata | [src/losses.py:566](src/losses.py#L566) — `DiceFocalGSLLoss` | Eq. 8 |
| DTM signed per slice (helper) | [src/dataset.py:115](src/dataset.py#L115) — `compute_signed_dtm` | Fig. 2 |
| Dataset con DTM | [src/dataset.py](src/dataset.py) — `BraTSDataset(return_dtm=True)` | — |
| Augmentation con DTM | [src/train_utils.py](src/train_utils.py) — `get_augmentation(with_dtm=True)` | — |
| Training loop GSL | [src/train_utils.py:583](src/train_utils.py#L583) — `train_one_epoch_gsl` | Eq. 8 |
| Eval loop GSL | [src/train_utils.py:674](src/train_utils.py#L674) — `evaluate_gsl` | — |
| Script offline (pesi + DTM) | [scripts/precompute_gsl_stats.py](scripts/precompute_gsl_stats.py) | Eq. 13, Fig. 2 |

---

## 2. Mappatura formula → codice

### 2.1 GSL (Eq. 12)
$$\mathcal{L}_{gsl} = 1 - \frac{\sum_{k} w_k \sum_{i} \big(D_i^k\,(1 - (T_i^k + P_i^k))\big)^2}{\sum_{k} w_k \sum_{i} (D_i^k)^2}$$

- $P_i^k$ = `softmax(logits)` (applicata internamente).
- $T_i^k$ = one-hot del ground-truth.
- $D_i^k$ = **DTM signed del GT** (positiva fuori, negativa dentro, 0 sul bordo — Fig. 2), pre-calcolata e fornita dal DataLoader.
- $w_k$ = pesi globali iniettati (buffer del modulo).

**Proprietà chiave (verificata numericamente):** la GSL è **bounded in [0,1]**. Predizione perfetta $P=T$ → loss $=0$; "worst case" $P=1-T$ → loss $=1$ (Eq. 11). Il denominatore (solo $D$, indipendente dalla predizione) è la normalizzazione che mantiene la GSL sulla stessa scala della region loss — la proprietà che il paper cerca rispetto alla Boundary Loss non bounded (§2.1).

### 2.2 Pesi globali $w_k$ (Eq. 13)
$$w_k = \left(\frac{1}{\sum_{j} 1/N_j}\right)\frac{1}{N_k}$$

- $N_k$ = numero **totale** di voxel della classe $k$ su **tutto il dataset** (split train), calcolato offline da `count_voxels_per_class`.
- **Sono globali e costanti**, iniettati nella GSL — *non* ricalcolati per batch. Questo è esattamente ciò che distingue la GSL dalla Generalized Dice Loss (i cui pesi cambiano per batch, alterando il problema di ottimizzazione e causando oscillazioni del gradiente — §1.1.1, §2.1). **Soddisfa il requisito 3.3 della consegna.**

### 2.3 Schedulazione di $\alpha$ (Eq. 14–16)
Convenzione: $t$ = indice epoca 0-based, $T$ = epoche totali. $\alpha(0)=1$ (solo region), decresce a $0$ nell'epoca finale.

- **Linear** (Eq. 14): $\alpha = 1 - t/(T{-}1)$
- **Step** (Eq. 15): $\alpha = 1 - \lfloor t/h \rfloor / N_h$, con $N_h=\lfloor (T{-}1)/h \rfloor$ — **default del progetto, `h=5`**
- **Cosine** (Eq. 16): $\alpha = \tfrac12(1+\cos(\pi t/(T{-}1)))$

**Step scheduler (requisito della consegna):** mantiene $\alpha$ costante per $h$ epoche, permettendo ad AdamW di stabilizzarsi su ogni sotto-obiettivo prima che il target del gradiente cambi (§4). Con `h=5` su 30 epoche, $\alpha$ segue: `[1.0×5, 0.8×5, 0.6×5, 0.4×5, 0.2×5, 0.0×5]`.

> **Schedule disponibili:** tutte e tre (`linear`/`step`/`cosine`) sono implementate e selezionabili; `step h=5` è solo il default scelto. Il paper trova `step-5` migliore per Tumour Core/LiTS, `step-25` per Whole Tumour, `linear` per ET (Tab. 2) — quindi il valore ottimale dipende dal target; resta un iperparametro sweepabile.

### 2.4 DTM signed (Fig. 2)
`compute_signed_dtm` calcola, per ogni classe $k$: `D = edt(~B) - edt(B)` (B = maschera binaria). Positiva fuori, negativa dentro, 0 sul bordo. Classe assente nella slice → DTM tutta a zero (contributo nullo alla GSL, corretto: nessuna superficie da penalizzare).

---

## 3. Come usarla — pipeline end-to-end (4 passi)

### Passo 1 — Pre-calcolo offline (una volta sola)
```bash
# Pesi + DTM per slice
python -m scripts.precompute_gsl_stats
# ...oppure pesi + DTM + impacchettamento per evitare il collo di bottiglia I/O su Drive
python -m scripts.precompute_gsl_stats --archive tar
```
Produce:
- `processed_dataset/gsl_class_weights.json` — i pesi $w_k$ (Eq. 13) dal train set.
- `processed_dataset/<split>/dtms/<subject>_slice<idx>.npy` — DTM `[C,H,W]` per slice.
- `processed_dataset/<split>_dtms.{tar,zip}` — un archivio per split (con `--archive`).

È idempotente (salta i file DTM già presenti) → resumable su Colab. Evita il ricalcolo per-epoca delle DTM che rende costosa la Hausdorff Loss (§1.1.2). **Su Colab: copia l'archivio su `/content/` (NVMe), scompattalo lì e punta `BraTSDataset` al percorso locale — mai a Drive** (vedi §6 e il docstring dello script).

### Passo 2 — Dataset e DataLoader con DTM
```python
import json
from src.dataset import BraTSDataset
from src.train_utils import get_augmentation, make_generator, seed_worker, set_seed
from torch.utils.data import DataLoader

set_seed(42, deterministic=True)

augment = get_augmentation(p=0.5, with_dtm=True)          # DTM trasformata con image/mask
train_ds = BraTSDataset("processed_dataset/train", augment=augment, return_dtm=True)
val_ds   = BraTSDataset("processed_dataset/val",   augment=None,    return_dtm=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True,
    worker_init_fn=seed_worker, generator=make_generator(42),
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        worker_init_fn=seed_worker)
```
Ora ogni batch è una **tripla `(image, mask, dtm)`**.

### Passo 3 — Costruire la loss
```python
import json
from src.losses import DiceFocalGSLLoss, AlphaScheduler

with open("processed_dataset/gsl_class_weights.json") as f:
    wk = json.load(f)["weights"]                          # pesi globali Eq. 13

criterion = DiceFocalGSLLoss(
    num_classes=4,
    gsl_class_weights=wk,
    scheduler=AlphaScheduler(schedule="step", total_epochs=TOTAL_EPOCHS, step_length=5),
    dice_weight=1.0, focal_weight=1.0, gamma=2.0,
    ignore_background=True,
).to(DEVICE)
```

### Passo 4 — Training loop
```python
from src.train_utils import train_one_epoch_gsl, evaluate_gsl

for epoch in range(1, TOTAL_EPOCHS + 1):
    train_metrics = train_one_epoch_gsl(
        model, train_loader, criterion, optimizer, DEVICE,
        epoch=epoch - 1,                  # 0-based for the scheduler
        crop_size=CROP_SIZE, scaler=scaler, amp_dtype=AMP_DTYPE,
    )
    val_metrics = evaluate_gsl(model, val_loader, criterion, DEVICE, crop_size=CROP_SIZE)
    # train_metrics ora include: loss, region_loss, gsl_loss, alpha, dice_<class>
    print(f"epoch {epoch}  alpha={train_metrics['alpha']:.2f}  "
          f"region={train_metrics['region_loss']:.4f}  gsl={train_metrics['gsl_loss']:.4f}")
```

> `criterion.set_epoch(epoch)` è chiamato **una volta per epoca** dentro `train_one_epoch_gsl` → tutti i batch dell'epoca usano lo stesso $\alpha$ (nessun cambio per-batch del problema di ottimizzazione).

---

## 4. Scelte di design e fedeltà al paper

1. **Region loss = Dice-Focal (non Dice-CE).** Il paper usa Dice-CE come $\mathcal{L}_{region}$ ma nota esplicitamente (Discussion) che si possono usare Dice puro o **focal loss**. Riusiamo la `CombinedLoss` (Dice+Focal) già esistente e validata del progetto — scelta legittima e allineata. La consegna chiede infatti $\mathcal{L}_{DiceFocal}$.

2. **2D slice-based, non 3D nnU-Net.** Il paper usa patch 3D 128³; noi restiamo nel paradigma 2D del progetto. La DTM è quindi calcolata **per slice 2D**, coerente con la pipeline (le slice `.npy` sono 2D). Questo non cambia la formula GSL, solo la dimensionalità di $D$.

3. **DTM pre-calcolata + augmentation rigida.** La DTM è generata offline (efficienza) e passata ad albumentations come *additional target* di tipo `image`, così subisce le stesse trasformazioni spaziali della maschera. In modalità `with_dtm=True` la pipeline è **ristretta a sole isometrie rigide** (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`): sotto queste trasformazioni la DTM resta una mappa di distanza **esatta** (nessuna interpolazione). `ShiftScaleRotate` ed `ElasticTransform` sono **categoricamente escluse** in questa modalità perché altererebbero le distanze euclidee reali e, via interpolazione, inquinerebbero il gradiente "chirurgico" della GSL. Evita anche il ricalcolo per-batch.

4. **Pesi globali iniettati.** Conformi a Eq. 13, calcolati una volta sul train set, costanti durante il training — il punto centrale che il paper contrappone alla GDL.

5. **Retrocompatibilità.** `BraTSDataset` con `return_dtm=False` (default) e `train_one_epoch`/`evaluate` restano **invariati**: i tre modelli baseline (Blocco 1/2) continuano a girare con `CombinedLoss` senza modifiche. La pipeline GSL è una variante aggiuntiva, attivata solo dai flag/funzioni dedicati.

6. **Determinismo + AMP.** Le funzioni GSL riusano gli stessi accorgimenti del Blocco 1: loss in fp32 sotto autocast (stabilità), supporto bf16, grad clipping sui soli parametri allenabili, seeding deterministico.

---

## 5. Verifiche eseguite

Tutte le verifiche numeriche sono passate (in locale, con `torch`/`numpy`/`scipy`; le dipendenze ML pesanti non sono installate, coerente col fatto che il training gira su Colab):

- **Eq. 13:** $w_k$ esattamente proporzionali a $1/N_k$, normalizzati (somma 1), classe più rara → peso massimo.
- **Eq. 14–16:** $\alpha(0)=1$, monotòno decrescente a 0, bounded [0,1]; step costante per blocco di $h$ epoche; `step h=5` produce i 6 gradini attesi.
- **Eq. 12 (GSL):** $P=T \Rightarrow 0$, $P=1{-}T \Rightarrow 1$ (bound esatti); modulo combacia col calcolo manuale; bounded [0,1] su input random.
- **DTM (Fig. 2):** segno corretto (+fuori / −dentro / 0 bordo); classi assenti → DTM nulla.
- **Eq. 8 (combinata):** epoca 0 → solo region ($\alpha=1$); epoca finale → solo GSL ($\alpha=0$); epoche intermedie → combinazione convessa esatta; gradiente fluisce.
- **Loop end-to-end:** simulazione con DataLoader fittizio (triple con DTM), center-crop DTM/image/mask allineato, $\alpha$ che evolve 1.0→0.0, forward/backward stabili.

---

## 6. Note operative

- **Costo storage DTM e I/O su Colab (CRITICO):** una DTM `[4,240,240]` float32 ≈ 0.9 MB/slice → ~35 GB per 39.538 slice, frammentati in decine di migliaia di file. **Leggere file `.npy` minuscoli direttamente da Google Drive durante il training affama la A100** (latenza per-file). Workflow corretto:
  1. Generare le DTM su Drive impacchettandole: `python -m scripts.precompute_gsl_stats --archive tar` (o `zip`) → un **singolo archivio per split** su Drive.
  2. Su Colab copiare l'archivio da Drive allo **storage locale NVMe** (`/content/`) — una copia sequenziale, non migliaia di letture piccole — e scompattarlo lì.
  3. Puntare `BraTSDataset(..., return_dtm=True, dtm_dir="/content/processed_dataset/<split>/dtms")` al **percorso locale**, **MAI** a Drive (vale anche per images/masks).

  `tar` = non compresso (estrazione più rapida); `zip` = deflate leggero (footprint/copia minori). In alternativa si può ridurre la DTM a `float16` se lo spazio stringe. Dettagli nel docstring di [scripts/precompute_gsl_stats.py](scripts/precompute_gsl_stats.py).
- **Coerenza con Fase 2:** se in Fase 2 si porta `crop_size=240`, la DTM (calcolata a 240) e il crop combaciano senza perdite. Con crop<240 la DTM viene croppata dalla stessa `compute_crop_offsets` (allineata a image/mask).
- **Sweep schedule:** per riprodurre l'analisi del paper (Tab. 2), basta cambiare `AlphaScheduler(schedule=..., step_length=...)` senza toccare il resto.
- **Background nella GSL:** `GeneralizedSurfaceLoss(include_background=True)` di default somma su tutte le classi (come il paper). Si può passare `gsl_include_background=False` in `DiceFocalGSLLoss` per concentrarsi sulle superfici foreground.
