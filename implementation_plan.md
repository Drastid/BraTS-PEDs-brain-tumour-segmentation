# IMPLEMENTATION PLAN — Roadmap Esecutiva Pre-Colab (Train-Once)

**Progetto:** Segmentazione di Tumori Cerebrali Pediatrici (BraTS-PEDs)
**Riferimento SOTA:** Cariola et al., *Scientific Reports* (2025) 15:22595 — vedi `confronto_sota.md`
**Documento prodotto/aggiornato:** 2026-06-21

---

## COME USARE QUESTO DOCUMENTO

> Questo file è una **roadmap esecutiva da seguire passo-passo**, scritta per essere letta in una **nuova finestra/contesto** insieme a `context.md` e a tutti i file di progetto (`src/`, `notebooks/`). Funziona come un secondo `context.md`: descrive *cosa modificare, dove, e in che ordine* **prima** di lanciare la pipeline su Google Colab (A100 ~40 GB).

**Scenario operativo:**
- **NON ho pesi/checkpoint.** Tutto verrà **ri-addestrato da zero su Colab**.
- **Obiettivo primario:** sistemare l'intero progetto **in locale** (modifiche al codice, configurazioni, logging) **prima** di spostarsi su Colab, in modo da **addestrare UNA VOLTA SOLA** e ottenere tutto il necessario per le analisi.
- **Paradigma invariato:** resto in **2D slice-based**. Nessuna conversione a 3D. Nessuno stravolgimento.
- **Cross-domain su adulti: ABBANDONATO.** Focus sulla **robustezza interna**.
- **Hardware target:** A100 ~40 GB (Colab). Sblocca batch grande, crop ampio, 5-fold e sweep economici.

**Principio guida — la regola "train-once":**
La domanda non è "quale esperimento ha il miglior ROI", ma **"cosa deve essere già nel codice PRIMA di premere train, così da non dover ri-addestrare?"**. Da qui la divisione netta del documento:

- **FASE A — Modifiche PRE-TRAINING (da fare in locale, ora).** Tutto ciò che entra nel grafo di training: loss, architetture, dataset/CV, config, logging. Se manca qui, va rifatto il training.
- **FASE B — Campagna di training su Colab (train-once).** Cosa lanciare e cosa salvare.
- **FASE C — Analisi POST-TRAINING (sui checkpoint/predizioni salvati, no ri-training).** Post-processing, ensemble, test statistici, incertezza.

⚠️ **L'errore da evitare:** lasciare per "dopo" qualcosa che richiede di essere nel training (es. deep supervision, una nuova loss, il logging per-soggetto). Verrebbe scoperto solo a training finito → ri-addestramento completo. **La FASE A serve esattamente a prevenirlo.**

---

## STEP 0 — INVENTARIO E VERIFICA (fare per primo, in locale)

Prima di toccare qualsiasi cosa, **verificare lo stato reale del codice** rispetto a quanto descritto in `context.md`. Il codice dovrebbe essere presente; questo step conferma che file e firme corrispondano.

**Checklist di verifica (leggere questi file e annotare cosa esiste davvero):**

| File atteso (da context.md) | Cosa verificare | Stato |
|---|---|---|
| `src/dataset.py` | `BraTSDataset`, `load_subject()`, `clip_outliers()`, `zscore_normalise()`, `get_valid_slice_indices()` | ☐ |
| `src/losses.py` | `DiceLoss`, `FocalLoss`, `CombinedLoss` (firme e parametri) | ☐ |
| `src/train_utils.py` | `get_augmentation`, `get_class_weights`, `MetricTracker`, `center_crop`, `train_one_epoch`, `evaluate`, `save/load_checkpoint` | ☐ |
| `src/models.py` | wrapper SegFormer 4-canali, eventuali factory dei modelli | ☐ |
| `src/eval_utils.py` | `predict_volume`, `compute_dice_volume`, `compute_hd95_volume`, `remove_small_components` | ☐ |
| `src/evaluate_3d_test.py` | CLI 3D eval (`--model unet|fpn|segformer|all`) | ☐ |
| `notebooks/0X_*.ipynb` | notebook di preprocessing/training/eval | ☐ |
| `split.json` | split paziente-level esistente (Strategy C) | ☐ |
| `requirements.txt` | versioni pacchetti | ☐ |

**Azione:** se un file/funzione manca o ha firma diversa da `context.md`, **annotarlo qui sotto** prima di procedere. Tutte le modifiche delle fasi successive assumono questi file come base.

> **Note di discrepanza (da compilare durante la verifica):**
> - _(es. "losses.py non ha CombinedLoss, solo Dice e Focal separate")_
> - _…_

---

## STEP 1 — SETUP COLAB-READY (infrastruttura, pre-training)

Modifiche che rendono il progetto eseguibile e resiliente su Colab. **Nessuna di queste cambia i risultati**, ma senza di esse il train-once su Colab è fragile.

### 1.1 — Persistenza su Google Drive
- Aggiungere all'inizio del notebook/script di training il **mount di Drive** e puntare lì le cartelle che devono sopravvivere alle disconnessioni:
  - `checkpoints/` → Drive
  - `evaluation_outputs/` → Drive
  - `processed_dataset/` → Drive (per non ri-estrarre i 39.538 `.npy` ad ogni sessione)
  - `logs/` (nuovo) → Drive
- **Dove:** cella di setup iniziale + un file `src/paths.py` (nuovo, opzionale) che centralizza i path con un flag `ON_COLAB`.

### 1.2 — Resume da checkpoint robusto
- Verificare/estendere `save_checkpoint`/`load_checkpoint` (`src/train_utils.py`) perché salvino **anche**: epoca corrente, stato optimizer, stato scheduler, stato AMP scaler, RNG state, e **fold corrente** (per la CV).
- Il loop di training deve poter **riprendere dall'ultima epoca/fold** se la sessione cade.
- **Salvataggio periodico:** salvare `last.pth` ogni epoca (non solo `best.pth`).

### 1.3 — Verifica hardware e config batch
- Cella che esegue `nvidia-smi` e **asserisce A100** (warning se T4/L4 fallback).
- Centralizzare gli iperparametri in un **`config.py` / dict di config** (nuovo) — vedi STEP 4. Batch size, crop, LR, epoche, loss devono essere parametri, non valori hard-coded sparsi.

### 1.4 — Determinismo e seed
- Fissare seed (=42) per `torch`, `numpy`, `random`, e impostare `torch.backends.cudnn` per riproducibilità. Necessario perché le ablazioni (STEP 5) siano confrontabili.

---

## STEP 2 — MODIFICHE ALLA LOSS (PRE-TRAINING, ad alto impatto)

⚠️ **Tutto in questo step DEVE essere nel codice prima del training.** Le loss non si possono cambiare a posteriori.

### 2.1 — Generalized Dice Focal Loss (GDFL) come opzione selezionabile
- **File:** `src/losses.py`.
- **Azione:** aggiungere una classe `GeneralizedDiceFocalLoss` (o integrare MONAI `GeneralizedDiceFocalLoss` se si aggiunge `monai` a `requirements.txt`).
- **Razionale:** la Generalized Dice pesa le classi per inverso del volume² → peso adattivo più forte su ET/NCR rari. Il paper mostra guadagno specifico sul recall ET.
- **Integrazione:** la scelta della loss deve essere un **parametro di config** (`loss = "combined" | "gdfl"`), così da addestrare entrambe le varianti nella stessa campagna (vedi STEP 6) **senza modifiche al codice tra un run e l'altro**.

### 2.2 — Boundary / Hausdorff-aware loss term
- **File:** `src/losses.py`.
- **Azione:** aggiungere un termine **boundary loss** (distance-transform based) combinabile con Dice/Focal tramite peso `boundary_weight` in config.
- **Razionale:** riporto HD95 ma non lo ottimizzo direttamente. Allineare il training alla mia metrica firma. Cfr. Kharraji et al. (Hausdorff loss in nnU-Net pediatrico).
- **Nota:** prevedere `boundary_weight=0.0` come default → la baseline resta confrontabile; il termine si attiva solo nelle config dedicate.

### 2.3 — Pesi Dice/Focal parametrizzati
- **File:** `src/losses.py` + config.
- **Azione:** rendere `dice_weight`/`focal_weight` (oggi 1:1) e gli `α` per-classe **parametri di config**, non costanti. Abilita lo sweep economico su A100 (STEP 6) senza editare codice.

---

## STEP 3 — MODIFICHE ARCHITETTURALI (PRE-TRAINING)

⚠️ Anche queste entrano nel grafo: vanno decise prima del training. Per il **train-once** servono come **varianti di modello selezionabili da config**, così da addestrarle tutte nella stessa campagna.

### 3.1 — Deep Supervision sul decoder (U-Net/FPN)
- **File:** `src/models.py` (wrapper) + `src/train_utils.py` (loop che somma le loss ausiliarie).
- **Azione:** esporre le feature dei livelli intermedi del decoder (`smp` lo consente), aggiungere head ausiliarie, sommare loss con pesi decrescenti.
- **Razionale:** stabilizza il gradiente sulle classi rare e mitiga il "click" ritardato (epoca 15 U-Net). Tecnica SOTA standard BraTS.
- **Config flag:** `deep_supervision: bool`.

### 3.2 — Attention gates sulle skip connections (Attention U-Net)
- **File:** `src/models.py`.
- **Azione:** variante U-Net con attention gate sulle skip. Selezionabile come `arch = "att_unet"`.
- **Razionale:** sopprime regioni cerebrali sane irrilevanti, concentra la capacità sui tumori piccoli (difficoltà pediatrica). Allineato all'attention del paper, ma sul contesto spaziale.

### 3.3 — Input 2.5D / multi-slice (opzionale)
- **File:** `src/dataset.py` (DataLoader: stack di N slice adiacenti come canali) + adattamento canali di input dei modelli.
- **Azione:** opzione `n_adjacent_slices: int` (1 = comportamento attuale, 3/5 = 2.5D).
- **Razionale:** recupera contesto inter-slice (gap 2D→3D del 20–25%) **restando in 2D**. **Non più dettato dalla memoria** (con A100 non è un workaround), ma valido come miglioria.
- **Priorità:** opzionale per il primo train-once; se incluso, va deciso ORA perché cambia gli input channel.

> **Modelli base confermati (invariati):** U-Net/ResNet34, FPN/ResNet34, SegFormer-B1/MiT-B1, tutti ImageNet-pretrained, patch-embedding 4-canali per SegFormer come in `context.md`.

---

## STEP 4 — DATASET, SPLIT E CROSS-VALIDATION (PRE-TRAINING)

⚠️ La struttura della CV determina **quali** modelli alleni e su **quali** dati: va fissata prima del training, altrimenti i fold non sono riproducibili.

### 4.1 — Generazione dei fold 5-Fold Stratified (paziente-level)
- **File:** `notebooks/02_preprocessing.ipynb` o nuovo `src/make_folds.py`.
- **Azione:** generare **5 fold stratificati a livello soggetto**, stratificando — come in Strategy C — per **presenza ET + quartile di volume tumorale**. Salvare `folds.json` (lista di subject_id per ogni fold).
- **Razionale:** con 26 soggetti di test la varianza è alta (std Dice ET ~0.43). Il 5-fold dà intervalli di confidenza credibili ed è il **pilastro della robustezza interna** richiesta. Con A100 il costo "5× training" è gestibile.
- **Compatibilità:** mantenere anche `split.json` esistente per la baseline single-split, così i risultati restano confrontabili con `context.md`.

### 4.2 — Crop e batch parametrizzati
- **File:** `src/train_utils.py` (`center_crop`) + config.
- **Azione:** `crop_size` da config (default 192; abilitare 224 o 240-full grazie all'A100). `batch_size` da config (16 → 64–128).
- **Razionale:** ABL-8 (crop) e ABL-9 (batch) diventano pienamente fattibili; nessun taglio dettato dalla VRAM.
- ⚠️ **Ricalibrare il LR** per batch grande (linear/sqrt scaling) — prevedere `lr` in config in funzione di `batch_size`.

### 4.3 — Config centralizzata (chiave del train-once)
- **File:** nuovo `src/config.py` o `configs/*.yaml`.
- **Azione:** un singolo punto che definisce **un esperimento** = {arch, loss, deep_supervision, attention, n_adjacent_slices, crop_size, batch_size, lr, fold, seed, …}.
- **Razionale:** permette di lanciare l'intera **griglia di esperimenti** (modelli × loss × ablazioni × fold) da config, **senza editare codice tra un run e l'altro**. È ciò che rende possibile addestrare tutto in una campagna unica e riproducibile.

---

## STEP 5 — LOGGING E SALVATAGGIO (PRE-TRAINING, critico per le analisi post-hoc)

⚠️ **Questo è lo step più sottovalutato e il più pericoloso da dimenticare.** Le analisi della FASE C (test statistici, ensemble, incertezza, post-processing) sono possibili *solo* se durante il train-once si salvano gli artefatti giusti. Se non li salvo, devo ri-addestrare.

### 5.1 — Salvare le PREDIZIONI volumetriche, non solo le metriche
- **File:** `src/evaluate_3d_test.py` / loop di eval.
- **Azione:** per ogni modello e ogni soggetto di test/val, salvare la **maschera predetta 3D** (e/o le **probabilità per-classe**, necessarie per l'ensemble pesato) su Drive.
- **Perché:** ensemble (C.2), post-processing ET/WT (C.1) e derivazione TC/WT/ET (C.4) lavorano su queste predizioni. Senza, niente analisi post-hoc.

### 5.2 — Salvare le metriche PER-SOGGETTO (non solo le medie)
- **File:** eval loop.
- **Azione:** produrre un CSV `per_subject_metrics_{model}.csv` con Dice/HD95 per soggetto e per classe (NCR/ED/ET **e** TC/WT/ET).
- **Perché:** i **test statistici** (Friedman/Conover, C.3) richiedono i valori per-soggetto. È l'evoluzione del `per_subject_fg_dice_all_models.csv` citato in `context.md`.

### 5.3 — Salvare logit/probabilità per MC-Dropout
- **Azione:** assicurarsi che i modelli con dropout mantengano i layer di dropout accessibili in inference (per attivarli a posteriori senza ri-training).
- **Perché:** MC-Dropout (C.5a) richiede solo forward pass ripetuti su pesi esistenti — ma solo se il dropout è nel modello allenato.

### 5.4 — Logging strutturato della campagna
- **Azione:** un `runs_registry.csv` (o W&B/TensorBoard su Drive) che per ogni run registra: config completa, best epoch, metriche, path del checkpoint e delle predizioni.
- **Perché:** con decine di run (modelli × loss × fold × ablazioni) serve tracciabilità per non perdersi.

---

## STEP 6 — DEFINIZIONE DELLA CAMPAGNA DI TRAINING (cosa si addestra in train-once)

Matrice degli esperimenti da addestrare **in un'unica campagna** su Colab. Ogni riga = un run definito da config (STEP 4.3). Ordinata per priorità: se il tempo GPU stringe, fermarsi alla priorità raggiunta lascia comunque un risultato completo.

| Run | Modello | Loss | Extra | Fold | Scopo |
|---|---|---|---|---|---|
| **R1** | SegFormer-B1 | Combined (Dice+Focal) | baseline | 5-fold | Baseline robusta (modello migliore) |
| **R2** | U-Net/ResNet34 | Combined | baseline | 5-fold | Baseline + membro ensemble |
| **R3** | FPN/ResNet34 | Combined | baseline | 5-fold | Baseline + membro ensemble |
| **R4** | SegFormer-B1 | **GDFL** | ET-focused | 5-fold | Variante ET (per ensemble pesato) |
| **R5** | SegFormer-B1 | Combined | **+ boundary** | 5-fold | Ottimizzazione HD95 |
| **R6** | SegFormer-B1 | Combined | **deep supervision** | 5-fold | Stabilità + classi rare |
| **R7** | SegFormer-B1 | Combined | **2.5D (3 slice)** | 5-fold | Recupero contesto inter-slice (opz.) |
| **R8** | Att-U-Net | Combined | attention gates | 5-fold | Variante architetturale |

**Ablazioni (STEP 2-4 già in codice → solo cambio config):** ABL-1 (no Focal), ABL-2 (no pretrain), ABL-3 (no two-phase), ABL-4/5 (augmentation), ABL-6 (no clip), ABL-7 (no post-proc → in eval), ABL-8 (crop 240), ABL-9 (batch). Eseguibili come run aggiuntivi **sul solo SegFormer** + conferma delle 2-3 più impattanti su U-Net/FPN.

> **Strategia anti-spreco GPU:** addestrare prima **R1–R4 a 5 fold** (coprono baseline + ensemble + ET). R5–R8 e le ablazioni possono partire da meno fold (es. 1–2) per uno screening, e completare a 5 fold solo le varianti che mostrano guadagno. Questo è il bilanciamento corretto tra train-once completo e budget GPU finito.

**Training config invariata (da context.md, parametrizzata):** two-phase (5 ep freeze @LR 3e-4 → 25 ep full, enc 1e-5 / dec 1e-4), CosineAnnealingLR per fase, weight decay 1e-4, AMP, grad clip 1.0, augmentation geometrica-only.

---

## FASE C — ANALISI POST-TRAINING (sui checkpoint/predizioni salvati, NO ri-training)

Tutto qui si esegue **dopo** la campagna, su artefatti salvati nello STEP 5. Nessun ri-addestramento. Si può fare anche in locale/CPU scaricando le predizioni da Drive.

### C.1 — Post-processing ET/WT-ratio
- **File:** `src/eval_utils.py`.
- **Azione:** azzerare ET se ET/WT < τ; sweep τ ∈ [0.005, 0.05] sul val, applicare τ* sul test. (Paper: τ=0.02.)
- **Richiede:** maschere/probabilità salvate (5.1). **Impatto:** alto su Dice/HD95-ET.

### C.2 — Ensemble dei modelli
- **(a) Deep Ensemble naturale:** media delle predizioni di R1/R2/R3 → guadagno Dice + mappa di disagreement.
- **(b) Ensemble pesato per-regione (come il paper):** pesi TC/WT/ET via **random search sul val** sopra le probabilità di R1/R2/R3/R4 (incluso il modello GDFL, forte su ET).
- **Richiede:** probabilità per-classe salvate (5.1). **Impatto:** alto su ET/WT (nel paper, significativo).

### C.3 — Test statistici di significatività
- **File:** nuovo `notebooks/09_statistics.ipynb` o `src/stats.py`.
- **Azione:** Friedman test tra modelli + post-hoc Conover con correzione FDR sui Dice **per-soggetto**.
- **Richiede:** `per_subject_metrics_*.csv` (5.2). **Impatto:** alto sul rigore; distingue differenze reali dal rumore.

### C.4 — Derivazione regioni gerarchiche TC/WT/ET
- **Azione:** da NCR/ED/ET (predizioni native) costruire TC = NCR∪ET, WT = NCR∪ED∪ET; ricalcolare Dice/HD95.
- **Richiede:** maschere salvate (5.1). **Impatto:** rende i numeri **commensurabili con la SOTA** (vedi `confronto_sota.md §1.2`). **Da riportare per ogni esperimento.**

### C.5 — Uncertainty Estimation
- **(a) MC-Dropout:** T=50+ forward pass con dropout attivo → media + varianza per-voxel (A100/locale). Richiede dropout nei modelli (5.3).
- **(b)** La disagreement del Deep Ensemble (C.2a) è già una mappa di incertezza gratuita.
- **Impatto:** interpretabilità clinica sui bordi, collegata a HD95.

---

## SEQUENZA OPERATIVA COMPLETA (checklist da seguire)

**In locale (prima di Colab):**
1. ☐ **STEP 0** — Inventario codice, annotare discrepanze.
2. ☐ **STEP 1** — Setup Colab-ready (Drive paths, resume, seed, verifica HW).
3. ☐ **STEP 2** — Loss: GDFL + boundary + pesi parametrizzati in `losses.py`.
4. ☐ **STEP 3** — Architetture: deep supervision, attention U-Net, (2.5D opz.) come varianti da config.
5. ☐ **STEP 4** — Fold 5-fold (`folds.json`), crop/batch parametrizzati, `config.py` centralizzata.
6. ☐ **STEP 5** — Logging: salvare predizioni 3D + probabilità + metriche per-soggetto + registry.
7. ☐ **Smoke test locale** — 1 epoca su pochi soggetti per verificare che ogni config giri senza errori (CPU o GPU locale se disponibile). **Critico:** scopre i bug PRIMA di sprecare GPU su Colab.

**Su Colab (train-once):**
8. ☐ Mount Drive, verifica A100, (eventuale) estrazione/sync `processed_dataset/`.
9. ☐ **STEP 6** — Lanciare la campagna R1–R4 (5-fold) → poi R5–R8 + ablazioni (screening → completamento selettivo).
10. ☐ Verificare che predizioni e CSV per-soggetto siano salvati su Drive ad ogni run.

**Post-training (locale o Colab, no ri-training):**
11. ☐ **FASE C** — Post-proc ET/WT (C.1), ensemble (C.2), statistica (C.3), TC/WT/ET (C.4), incertezza (C.5).
12. ☐ Aggiornare `confronto_sota.md` con i numeri TC/WT/ET ottenuti.

---

## PRINCIPI DA NON DIMENTICARE

1. **Train-once:** se una modifica entra nel grafo di training (loss, architettura, input channel, dataset/fold), **DEVE essere in FASE A**. Scoprirlo dopo = ri-addestrare.
2. **Salva le predizioni, non solo le metriche** (STEP 5.1): è la condizione abilitante di metà della FASE C.
3. **Salva le metriche per-soggetto** (STEP 5.2): senza, niente test statistici.
4. **Tutto da config** (STEP 4.3): un solo punto di controllo → niente edit di codice tra run → riproducibilità.
5. **Smoke test prima di Colab** (passo 7): non bruciare ore A100 su bug banali.
6. **Paradigma 2D invariato, cross-domain rimosso:** robustezza interna (5-fold + statistica + ensemble + incertezza) come obiettivo, senza stravolgere il progetto.
