# Pipeline 3D — Log di avanzamento

Registro incrementale della migrazione 2D → 3D di BraTS-PEDs, in esecuzione
stop-and-go secondo `master/road_3D.md`. Ogni sezione corrisponde a un punto
della roadmap completato e approvato.

---

## Setup iniziale — Workspace `BraTS-PEDs-3D/`

- Creata la directory `BraTS-PEDs-3D/` allo stesso livello di `master/`, come
  progetto indipendente (non sotto `master/`).
- Alberatura iniziale:
  ```
  BraTS-PEDs-3D/
  ├── data/
  │   ├── raw/            (destinazione dataset NIfTI originale — da popolare manualmente)
  │   └── processed_3d/   (output preprocessing locale — Punto 2 roadmap)
  ├── src/                (codice — Dataset, modelli, training, eval 3D)
  ├── notebooks/          (EDA/preprocessing locali + notebook Colab)
  ├── weights/            (pesi pre-addestrati scaricati — SegResNet bundle, SwinUNETR SSL)
  ├── checkpoints/        (output training)
  ├── evaluation_outputs/ (metriche 3D)
  ├── tests/
  ├── .gitignore          (esclude data/, weights/, checkpoints/, evaluation_outputs/)
  └── pipeline3D.md        (questo file)
  ```
- Repo git indipendente inizializzato (`git init`), nessun legame con `master/`.
- `data/raw/` lasciata vuota: il dataset NIfTI originale ("PKG - BraTS-PEDs-v1")
  non è presente in questa working area — l'utente lo copierà manualmente.
- Nessun codice applicativo ancora scritto: solo scheletro di cartelle,
  `.gitignore`, `__init__.py` vuoti in `src/` e `tests/`.

**Prossimo passo**: Punto 1 della roadmap — verifica del bug di mappatura
label (§1.2) e fix a 5 classi (§1.3), applicato però da zero nel nuovo
progetto (non c'è ancora un `constants.py`/`dataset.py` qui: verranno creati
direttamente corretti a 5 classi, senza passare per lo stato buggato).

---

## Punto 1 — Verifica label + `constants.py`/`dataset.py` a 5 classi (roadmap §1)

- **Ambiente locale**: nessun Python locale aveva `nibabel`/`numpy`/`scipy`
  installati (né globale né un venv preesistente in `master/`). Creato un
  venv dedicato `BraTS-PEDs-3D/.venv` (escluso da git) con
  `nibabel numpy scipy tqdm scikit-learn`, isolato dall'ambiente di sistema.
- **Dataset raw**: confermata la presenza di `data/raw/BraTS-PEDs-v1/Training/`
  con 257 cartelle soggetto (copiate dall'utente).
- **Script di verifica** (`scripts/verify_label_mapping.py`, §1.2 roadmap):
  conta, per ciascuno dei 257 soggetti training, la combinazione di etichette
  uniche presenti nella maschera di segmentazione grezza.
  - **Risultato**: **27/257 soggetti (10.5%)** hanno **sia** la label 3 (CC)
    **sia** la label 4 (ED) co-presenti come regioni distinte nello stesso
    volume. Altri 56 soggetti hanno solo label 3, altri 25 solo label 4.
  - **Conclusione confermata**: la remap `seg = np.where(seg==4, 3, seg)`
    presente nel vecchio progetto 2D (`master/src/dataset.py`) fonde
    illegittimamente due regioni tumorali cliniche distinte (Cystic Component
    ed Edema) in un'unica classe per oltre un decimo dei soggetti. Il bug è
    quindi confermato empiricamente sui dati reali, non solo per lettura della
    letteratura BraTS-PEDs.
- **`src/constants.py`** (nuovo, da zero — nessuna remap):
  - `NUM_CLASSES = 5`
  - `CLASS_NAMES = ("background", "ET", "NET", "CC", "ED")` — convenzione
    pediatrica ufficiale, non quella adulti (`NCR/ED/ET`) usata nel vecchio
    progetto.
  - `VOXEL_FREQ` lasciato a `None` con commento esplicito: le frequenze reali
    a 5 classi andranno calcolate sul TRAIN split una volta pronto
    `split.json` (punto successivo della roadmap), non riciclate dal vecchio
    vettore a 4 classi (che sarebbe comunque sporco per via della remap).
- **`src/dataset.py`** (nuovo, da zero): **scelta concordata con l'utente** —
  questo modulo fa **solo I/O e validazione**, nessuna normalizzazione
  numerica (niente clip percentile, niente z-score). La normalizzazione vivrà
  esclusivamente nelle transform MONAI (`NormalizeIntensityd` o simili) nel
  training loop lato Colab, coerentemente con la dicotomia
  locale-preprocessing/cloud-training richiesta.
  - `get_subject_paths`, `validate_subject_files`: individuano/verificano i 5
    file NIfTI attesi (4 modalità + seg).
  - `load_subject_raw`: carica le 4 modalità + segmentazione come array NumPy
    grezzi (`[4,H,W,D]` float32 + `[H,W,D]` int16), **senza alcuna remap né
    normalizzazione**.
  - `validate_segmentation_labels`: verifica che le etichette osservate siano
    in `range(NUM_CLASSES)` — usata per controlli di integrità.
  - `load_subject_nifti_meta`: espone affine/header NIfTI per la futura
    ricostruzione delle predizioni come NIfTI validi.
- **Smoke test manuale**: caricato il soggetto `BraTS-PED-00001-000` con
  `load_subject_raw` → shape `(4, 240, 240, 155)` / `(240, 240, 155)`,
  etichette uniche `[0, 1, 2, 3, 4]`, validazione `labels valid: True`. Nessun
  errore, nessuna remap applicata.

**Cosa NON è stato fatto in questo punto** (rimandato ai punti successivi
della roadmap): nessuna estrazione/riorganizzazione dei volumi in
`data/processed_3d/`, nessuno split.json, nessun calcolo di `VOXEL_FREQ` reale,
nessun preprocessing/normalizzazione.

---

## Punto 2 (parte 1) — Generazione dello split train/val/test (roadmap §2)

- **Bivio affrontato**: il vecchio progetto ha già `master/split.json`
  (Strategia C: stratificato per presenza-ET + quartile tumour-burden, seed
  42, 257 soggetti). Però quello split calcola "ET presence" sulla vecchia
  label 3 fusa (post-remap 4→3), che mescolava il vero ET con parte di
  CC/ED. **Decisione dell'utente**: non riusare quello split — rigenerarlo da
  zero con le label 5-classi native, per essere metodologicamente corretto.
  Una volta consolidato, `split_3d.json` verrà anche usato per ri-valutare il
  vecchio progetto 2D con uno split coerente.
- **`scripts/generate_split.py`** (nuovo): replica fedelmente la Strategia C
  del vecchio `02_preprocessing.ipynb` (stessa formula: `combined_label =
  et_presence*4 + quartile`, stesso seed 42, stesso split 80/10/10 via
  `sklearn.train_test_split` stratificato), ma conta i voxel per classe sulle
  **label native**: `n_et` (label 1 pura), `n_net` (2), `n_cc` (3), `n_ed` (4).
  `tumour_burden = n_et+n_net+n_cc+n_ed`.
- **Eseguito su tutti i 257 soggetti reali** in `data/raw/`. Risultato notevole:
  con le label corrette, il **66.5% dei soggetti ha ET presente** (label 1
  pura), contro il **42.0%** riportato dal vecchio notebook 2D sulla label
  fusa — conferma indiretta che la vecchia label 3 mescolava classi diverse e
  sottostimava la prevalenza clinica reale di ET.
- **Split risultante**: Train=205, Val=26, Test=26 (stesse proporzioni del
  vecchio progetto). ET rate ben bilanciato tra split: 66.3% (train) / 69.2%
  (val) / 65.4% (test). Nessuna sovrapposizione tra i tre insiemi, tutti i 257
  soggetti coperti (verificato via controllo insiemistico).
- **Output**: `data/split_3d.json` — stessa struttura del vecchio
  `split.json` (train/val/test come liste di subject_id) più un campo
  `label_scheme` esplicito, cosi' è chiaro a colpo d'occhio che questo split
  usa la convenzione a 5 classi.

**Prossimo sotto-passo di questo stesso punto (§2 roadmap)**: riorganizzazione
dei volumi NIfTI in `data/processed_3d/{train,val,test}/` secondo lo split appena
generato (fase locale, solo riorganizzazione — nessuna estrazione di patch,
nessuna normalizzazione, coerente con la Dicotomia Locale/Cloud). Prima di
lanciarla è necessaria una decisione su formato/organizzazione (vedi domanda
successiva).

---

## Punto 2 (parte 2) — Riorganizzazione dei volumi NIfTI (roadmap §2, fase locale)

- **Bivio affrontato**: copia fisica vs symlink dei volumi da `data/raw/` a
  `data/processed_3d/{train,val,test}/`. **Decisione dell'utente**: copia
  fisica (più robusta/portabile per la successiva compressione verso Colab,
  a costo di raddoppiare lo spazio disco).
- **Notifica preventiva data all'utente** (come richiesto): dataset raw 33 GB,
  213 GB liberi su disco, operazione non distruttiva e reversibile
  (`data/raw/` resta intatto). Confermato prima di lanciare.
- **`scripts/reorganize_volumes.py`** (nuovo): legge `data/split_3d.json`,
  copia per ciascun soggetto i 5 file NIfTI grezzi (4 modalità + seg) da
  `data/raw/BraTS-PEDs-v1/Training/<subject_id>/` a
  `data/processed_3d/<split>/<subject_id>/`. **Nessuna estrazione di patch,
  nessuna normalizzazione, nessuna conversione di formato** — solo
  riorganizzazione, coerente con la Dicotomia Locale/Cloud (la
  normalizzazione vivrà nelle transform MONAI lato Colab, punto successivo
  della roadmap).
- **Eseguito in background** (~33 GB di I/O, alcuni minuti). Completato con
  successo (exit code 0).
- **Verifica di integrità post-copia**:
  - Conteggio cartelle/file per split: train 205 cartelle × 5 file = 1025,
    val 26×5=130, test 26×5=130 — tutti i conteggi coincidono esattamente con
    gli attesi.
  - Dimensione totale copiata: 25 GB in `data/processed_3d/`.
  - Controllo a campione (15 soggetti su 257, 5 split casuali per split):
    75 file NIfTI caricati con `nibabel`, tutti leggibili senza errori;
    verificato anche che le etichette nei file `-seg.nii.gz` campionati siano
    nel range atteso `{0,1,2,3,4}` — nessuna anomalia.
- **Output finale di questo punto**:
  ```
  data/
  ├── split_3d.json                          (257 soggetti: 205/26/26, 5-classi native)
  └── processed_3d/
      ├── train/<subject_id>/<subject_id>-{t1c,t1n,t2f,t2w,seg}.nii.gz   (205 soggetti)
      ├── val/...                                                        (26 soggetti)
      └── test/...                                                       (26 soggetti)
  ```

**Punto 2 della roadmap completato integralmente** (split + riorganizzazione).
Nessuna normalizzazione applicata in questa fase, coerente con la richiesta
dell'utente e con la Dicotomia Locale/Cloud: i volumi in `processed_3d/` sono
pronti per essere compressi/trasferiti su Colab, dove le `monai.transforms`
(caricamento, normalizzazione, patch sampling) opereranno on-the-fly durante
il training (prossimi punti della roadmap: §3 modelli 3D, §5 training loop).

---

## Punto 3 — `src/models3d.py`: architetture 3D + caricamento pesi (roadmap §3)

- **Ambiente locale esteso**: il venv leggero (nibabel/numpy/scipy) non aveva
  torch/MONAI. **Bivio posto all'utente**: installarli localmente (CPU-only)
  per poter scrivere E testare subito il codice con smoke test reali, oppure
  scrivere alla cieca basandosi solo sulla documentazione MONAI e verificare
  poi su Colab. **Decisione dell'utente**: installare in locale. Aggiunto al
  venv: `torch` (2.12.1, CPU-only — il primo tentativo dall'index ufficiale
  PyTorch ha fallito per un problema di DNS della rete locale, risolto
  installando dal PyPI standard), `monai` (1.6.0), `einops` (richiesto da
  SwinUNETR), `pytest`.
- **Bivio sui pesi pre-addestrati di SwinUNETR**: la roadmap menziona due
  fonti (SSL NVIDIA grezzo vs BrainSegFounder fine-tuned). **Decisione
  dell'utente**: non hardcodare nessuna delle due — rendere
  `load_pretrained_3d` completamente agnostico rispetto alla fonte, cosi' la
  scelta specifica si rimanda al momento del training su Colab.
- **`src/models3d.py`** (nuovo):
  - `ARCH_NAMES = ("unet", "fpn", "segformer")` — nomi degli slot mantenuti
    identici al vecchio progetto 2D per compatibilita' di convenzione con
    `run_pipeline`/naming dei checkpoint, pur restituendo architetture diverse.
  - `build_model_3d(arch, in_channels=4, num_classes=5, roi=(128,128,128))`:
    - `"unet"` → `DynUNet` (stile nnU-Net, 5 livelli, stride-2 × 4).
    - `"fpn"` → `SegResNet` (rimpiazzo della FPN 2D — nessun equivalente FPN
      3D drop-in in MONAI; SegResNet realizza la stessa idea di aggregazione
      multi-scala ed e' l'unico dei tre con un bundle BraTS pronto nel MONAI
      Model Zoo).
    - `"segformer"` → `SwinUNETR` (rimpiazzo del SegFormer 2D, transformer 3D
      SOTA con pesi pre-addestrati pubblici disponibili).
    - Tutti costruiti SEMPRE con la testa gia' a `num_classes=5` (pediatrico).
  - `load_pretrained_3d(model, ckpt_path, drop_prefixes=(), ...)`: funzione
    **generica e agnostica rispetto alla fonte del checkpoint** (bundle MONAI,
    pesi SSL, HuggingFace, o un training precedente di questo progetto).
    Confronta chiave per chiave nome+shape tra lo state_dict del checkpoint e
    quello del modello target; tiene solo i tensori compatibili; la head di
    output (tipicamente con num_classes diverso nei checkpoint pubblici
    adulti) viene automaticamente esclusa per shape mismatch e resta
    inizializzata da zero. Gestisce anche il prefisso `module.` (checkpoint
    salvati da DataParallel/DDP) e diverse convenzioni di annidamento
    (`state_dict`/`model`/dict grezzo).
- **Test** (`tests/test_models3d.py`, eseguiti su CPU locale):
  - Forward pass su tensore fittizio `[1,4,128,128,128]` per tutte e tre le
    architetture → output `[1,5,128,128,128]` verificato per DynUNet,
    SegResNet, SwinUNETR (4 test, tutti PASSED).
  - Verifica che `build_model_3d` sollevi `ValueError` per un nome arch non
    valido.
  - **Test manuale aggiuntivo di `load_pretrained_3d`**: creato un checkpoint
    fittizio di SegResNet a 3 classi (simula un bundle pre-addestrato adulto
    TC/WT/ET), caricato in un modello target a 5 classi. Risultato: 81/83
    tensori caricati correttamente, i soli 2 tensori della head
    (`conv_final.2.conv.{weight,bias}`, shape 3 vs 5 canali) correttamente
    saltati per shape mismatch, forward pass finale verificato a
    `[1,5,128,128,128]` — confermato che il backbone pre-addestrato si
    trasferisce mentre la head pediatrica resta allenabile da zero.

**Cosa NON è stato fatto in questo punto** (rimandato ai punti successivi):
nessun download di pesi reali (SSL NVIDIA / bundle MONAI / BrainSegFounder) —
la scelta della fonte per SwinUNETR e degli altri due modelli è rimandata al
momento del training su Colab; nessuna loss 3D (`src/losses.py` — punto
successivo §4); nessun training loop (§5).

---

## Punto 4 — Loss 3D, DTM on-the-fly, metriche HD95 (roadmap §4)

- **Direttiva articolata dall'utente** (4 richieste esplicite, non solo la
  scelta A/B della roadmap): (1) MONAI `DiceFocalLoss` come baseline sicura di
  default; (2) porting N-D delle loss custom del vecchio progetto in un nuovo
  `src/losses_3d.py`, includendo la GSL in 3D; (3) integrazione di
  `monai.metrics.HausdorffDistanceMetric(percentile=95)` nel loop di
  validazione/test, loggata **separatamente per le 4 sub-regioni pediatriche**
  (ET, NET, CC, ED); (4) flag `--loss {dice_focal,gsl}` in `run_pipeline_3d.py`.
- **Bivio sulla Distance Transform Map (DTM) 3D**: la DTM esatta (scipy EDT)
  gira solo su CPU; un'alternativa sarebbe un'approssimazione GPU-friendly
  (es. morfologia iterativa) per evitare round-trip CPU↔GPU. **Decisione
  dell'utente, con motivazione tecnica propria**: scipy EDT esatta, calcolata
  SOLO dopo il patch sampling (su una patch 128³, non sul volume intero).
  Motivazione dell'utente: la DTM e' calcolata sulla ground truth, che e' una
  costante (non serve differenziabilità), e spostarla su GPU consumerebbe
  VRAM inutilmente; il costo si nasconde comunque dietro l'asincronia dei
  worker CPU del DataLoader.
- **`src/transforms.py`** (nuovo):
  - `compute_signed_dtm_nd(mask, num_classes)`: generalizzazione N-D di
    `compute_signed_dtm` dal vecchio progetto 2D (la formula e già
    dimension-agnostica in scipy; cambia solo la shape gestita).
  - `ComputeDistanceMapd(MapTransform)`: transform MONAI da comporre **dopo**
    `RandCropByPosNegLabeld` nella pipeline di training (punto §5, non ancora
    scritta) — riceve la patch già croppata (tipicamente 128³) e aggiunge la
    chiave `"distance_map"` al dizionario del batch, cosi' la loss vi accede
    liberamente in `train_3d.py`.
- **`src/losses_3d.py`** (nuovo, porting N-D del vecchio `master/src/losses.py`):
  - `DiceLoss`, `FocalLoss`, `CombinedLoss`: stessa formula del vecchio
    progetto, ma le riduzioni sommano su `tuple(range(2, ndim))` invece di
    assumere fisso `(1,2)` — funzionano identicamente su tensori
    `[B,C,H,W]` (2D) e `[B,C,D,H,W]` (3D) senza alcuna modifica al codice.
  - `_one_hot_channel_first`: helper che generalizza
    `F.one_hot(...).permute(0,3,1,2)` (fisso 2D) a qualunque rank.
  - `GeneralizedSurfaceLoss`, `DiceFocalGSLLoss`, `AlphaScheduler`,
    `compute_global_class_weights`: porting N-D della GSL (Celaya et al.),
    stessa logica di scheduling alpha(t) del vecchio progetto.
- **`src/losses_monai.py`** (nuovo, pipeline di default): wrapper
  `DiceFocalLossWrapper`/`build_dice_focal_loss` attorno a
  `monai.losses.DiceFocalLoss`, con adattamento della convenzione target
  (`[B,*spatial]` senza canale, come nel resto della pipeline, invece della
  convenzione nativa MONAI `[B,1,*spatial]`).
  - **Bug trovato e corretto durante il testing**: MONAI richiede che
    `weight` abbia lunghezza pari al numero di classi EFFETTIVAMENTE mediate
    — se `include_background=False`, il peso della classe 0 va scartato
    PRIMA di passarlo a `DiceFocalLoss`, altrimenti solleva `ValueError`
    ("length of weight sequence..."). Non era ovvio dalla sola lettura della
    firma; scoperto dal test automatico (vedi sotto) e corretto in
    `build_dice_focal_loss`.
- **`src/metrics_3d.py`** (nuovo): `SegmentationMetrics3D` incapsula
  `monai.metrics.DiceMetric` + `HausdorffDistanceMetric(percentile=95)`,
  entrambe con `reduction="none"` per preservare il dettaglio per-classe.
  `aggregate_and_reset()` ritorna un dict con `dice_<classe>`/`hd95_<classe>`
  per ciascuna delle 4 sub-regioni foreground (ET, NET, CC, ED) — MAI un
  unico valore aggregato — più `dice_mean_fg`/`hd95_mean_fg` come riepilogo
  aggiuntivo (media che ignora esplicitamente eventuali NaN, per non far
  collassare l'intero riepilogo se una singola classe è assente dal volume).
- **Test eseguiti (16/16 passati)**:
  - `tests/test_losses.py` (10 test): `DiceLoss`/`FocalLoss`/`CombinedLoss`
    verificate ESPLICITAMENTE sia in 2D `(64,64)` sia in 3D `(32,32,32)` con
    lo stesso identico codice — prova diretta che la generalizzazione N-D
    funziona; pipeline GSL completa in 3D (`ComputeDistanceMapd` →
    `GeneralizedSurfaceLoss`); scheduling di `alpha(t)` in `DiceFocalGSLLoss`
    (verificato `alpha(0)=1`, `alpha(T-1)=0`); wrapper MONAI con backward()
    e verifica gradienti finiti; `compute_global_class_weights` (la classe
    più rara riceve il peso maggiore).
  - `tests/test_metrics_3d.py` (2 test): predizione perfetta → Dice=1,
    HD95=0 per tutte le classi; caso limite con 3 sub-regioni su 4 assenti
    dal volume → verificato che `dice_mean_fg`/`hd95_mean_fg` restino
    **finiti** (non NaN) nonostante l'assenza di classi.
  - **Nota di comportamento osservata**: MONAI restituisce `hd95=0.0` (non
    NaN) quando sia predizione che GT sono assenti per una classe in un
    batch — comportamento leggermente diverso dal vecchio progetto 2D (che
    con `medpy` restituiva NaN in quel caso specifico, si veda
    `master/src/eval_utils.py::compute_hd95_volume`). Non è un errore, ma va
    tenuto a mente nel confronto diretto 2D-vs-3D delle metriche HD95 nel
    report finale.
  - `tests/test_models3d.py` (4 test, dal punto precedente): rieseguiti come
    controllo di regressione, ancora tutti PASSED.

**Cosa NON è stato fatto in questo punto** (rimandato deliberatamente, come
concordato): il flag `--loss {dice_focal,gsl}` in `run_pipeline_3d.py` — quel
file non esiste ancora nella sua interezza (DataLoader MONAI, training loop,
sliding-window validation sono contenuto del Punto 5, §5 della roadmap); il
flag verrà aggiunto naturalmente lì, dove vive già la logica di scelta
ottimizzatore/scheduler.

---

## Punto 5 — Dataset MONAI, training loop 3D, LR differenziato, `run_pipeline_3d.py` (roadmap §5)

- **Tre bivii posti e risolti prima di scrivere codice**:
  1. **Normalizzazione**: MONAI ha `NormalizeIntensityd` ma non applica il
     clip degli outlier a P99.5. **Decisione dell'utente, con motivazione
     tecnica propria**: replicare ESATTAMENTE la logica del vecchio progetto
     2D (clip-P99.5 sui soli voxel non-zero, poi z-score sui voxel clippati,
     background forzato a 0 esatto) in una transform custom
     `ClipAndNormalizeNonZerod`, per mantenere comparabilità 1:1 col 2D e la
     robustezza clinica contro gli artefatti da scanner.
  2. **Struttura del training**: script `.py` puro (non un notebook Colab),
     per riusabilità/testabilità/diff puliti.
  3. **Optimizer**: LR differenziato (backbone pre-addestrato a LR ridotto,
     head nuova a LR pieno) + freeze opzionale del backbone, implementato
     subito invece di partire con un LR singolo.
- **`src/transforms_3d.py`** (nuovo): `ClipAndNormalizeNonZerod(MapTransform)`
  — applica canale per canale (le 4 modalità indipendentemente) esattamente i
  6 passaggi richiesti dall'utente (maschera >0, percentile 99.5 sulla
  maschera, clip, media/std sui voxel CLIPPATI, z-score, background a 0
  esatto). **Verificata numericamente** contro un'implementazione di
  riferimento in NumPy che replica alla lettera `clip_outliers` +
  `zscore_normalise` del vecchio progetto: risultati identici entro
  `atol=1e-4`. Verificato anche il fallback per canali degeneri (< 10 voxel
  non-zero → tutto zero, stessa soglia del vecchio progetto).
- **`src/dataset_3d.py`** (nuovo): pipeline MONAI completa per la fase Colab.
  - `build_subject_dicts`: converte `data/processed_3d/<split>/<subject_id>/`
    nel formato `{"image": [4 path], "label": path}` atteso da `LoadImaged`.
  - `build_train_transforms`: `LoadImaged` → `EnsureChannelFirstd` →
    `ClipAndNormalizeNonZerod` → `RandCropByPosNegLabeld` (patch sampling
    bilanciato tumore/background, `pos=1,neg=1`) → flip/rotazione random →
    `RandScaleIntensityd`/`RandShiftIntensityd` (augmentation z-score-safe) →
    (opzionale) `ComputeDistanceMapd` se `with_dtm=True` (ramo `--loss=gsl`).
  - `build_eval_transforms`: stessa normalizzazione ma **nessun patch
    sampling** — il volume intero passa così com'è a
    `sliding_window_inference` nel training loop.
  - `build_dataloaders_3d`: assembla train/val loader, con `CacheDataset`
    opzionale per il train set.
  - **Testato su dati REALI** (2 soggetti veri da `data/processed_3d/train/`):
    caricamento + trasformazione di un singolo item in ~8.6s (CPU locale;
    su Colab con più worker sarà nascosto dall'asincronia); shape verificate
    `image=[4,128,128,128]`, `label=[1,128,128,128]`, range di intensità
    coerente con la normalizzazione (`[-0.80, 4.42]`); **la patch campionata
    contiene tutte e 5 le label native `{0,1,2,3,4}`** (ulteriore conferma
    indiretta che le label a 5 classi sono integre lungo tutta la pipeline);
    batch DataLoader `[4,4,128,128,128]` corretto (batch_size=2 ×
    num_samples=2). Testato anche il ramo `with_dtm=True` su ROI 64³: shape
    `distance_map=[5,64,64,64]`, valori con segno coerenti (range
    `[-71.7, +84.1]`).
- **`src/optim_3d.py`** (nuovo): `split_backbone_head_params` (separazione
  per IDENTITÀ dei tensori, non per nome — stessa tecnica del vecchio
  progetto 2D), `set_backbone_trainable` (analogo 3D di
  `set_encoder_trainable`), `build_optimizer_3d` (AdamW a 2 param-group).
  - **Nomi dei moduli "head" verificati empiricamente** (non assunti):
    ispezionati i `named_children()` reali di ciascuna architettura MONAI
    costruita — confermato `output_block` (DynUNet), `conv_final`
    (SegResNet), `out` (SwinUNETR) — e codificati in `_HEAD_MODULE_NAMES`.
  - **9/9 test passati** su tutte e tre le architetture: split
    backbone/head disgiunto e completo, LR effettivamente differenziato nei
    param_group dell'optimizer, freeze/unfreeze del backbone con la head
    sempre allenabile.
- **`src/train_3d.py`** (nuovo): `train_one_epoch_3d` (ramo `dice_focal`),
  `train_one_epoch_gsl_3d` (ramo `gsl`, aggiorna `alpha(t)` una volta per
  epoca), `evaluate_3d` (**sempre** su volume intero via
  `sliding_window_inference`, mai su patch — indipendentemente dal ramo di
  training: la GSL è un termine di training, le metriche di validazione
  restano sempre Dice/HD95 via `src/metrics_3d.py`). Più `MetricTracker`,
  `save_checkpoint`/`load_checkpoint` (reimplementati qui, non importati da
  `master/`, per rispettare l'isolamento del workspace).
- **`run_pipeline_3d.py`** (nuovo, CLI): il **flag `--loss {dice_focal,gsl}`**
  richiesto, più `--pretrained {auto,none}` (ablation scratch-vs-pretrained,
  road_3D.md §7), `--warmup-freeze-epochs` (freeze opzionale del backbone),
  `--roi`, `--arch`, iperparametri di training. Assembla modello → (opzionale)
  caricamento pesi pre-addestrati → DataLoader → optimizer → loss → loop di
  epoche con salvataggio best/last checkpoint su `dice_mean_fg`.
- **Smoke test end-to-end REALE eseguito** (non solo unit test): creato un
  mini data-root con 2 soggetti train + 1 val veri (copiati da
  `data/processed_3d/`), poi lanciato l'intero `run_pipeline_3d.py` su CPU
  locale:
  - **Ramo `--loss dice_focal`** (arch=fpn, roi=32³, 1 epoca): completato
    senza errori. Training + validazione con sliding-window sul volume
    intero 240×240×155 (~3.5 min su CPU — atteso; sarà enormemente più
    rapido su A100). Checkpoint `best.pth`/`last.pth` salvati correttamente.
  - **Ramo `--loss gsl`** (arch=unet, roi=32³, 2 epoche): completato senza
    errori. DTM calcolata on-the-fly per patch, scheduling di alpha attivo,
    fallback automatico ai pesi uniformi quando `gsl_class_weights.json` non
    è ancora presente (verificato il warning corretto). Sliding-window
    validation e checkpoint anch'essi corretti.
  - Dice riportati bassissimi (~0.001-0.002) come atteso: 1-2 epoche, 2
    soggetti, nessun pretraining, ROI minuscolo — è una verifica
    **strutturale** (la pipeline gira senza eccezioni end-to-end), non di
    qualità del modello.
  - Dati e checkpoint temporanei dello smoke test rimossi al termine.
- **Suite di test completa**: 28/28 test passati (4 modelli + 10 loss + 2
  metriche + 3 transform + 9 optim), nessuna regressione.

**Cosa NON è stato fatto in questo punto** (rimandato ai punti successivi
della roadmap): nessun training reale su A100/Colab (richiede l'infrastruttura
cloud, fuori scope di questo workspace locale); nessun download di pesi
pre-addestrati reali (la fonte per SwinUNETR/SegResNet resta una decisione da
prendere al momento del training vero, come concordato al Punto 3); nessuna
valutazione 3D nativa su test set con export NIfTI (§6 della roadmap).

---

## Punto 6 — Valutazione 3D nativa sul test set + export NIfTI (roadmap §6)

- **Due bivii posti e risolti**:
  1. **Post-processing**: includere `remove_small_components` (rimozione
     componenti connesse piccole, il "confetti effect" del vecchio progetto
     2D) anche in 3D, pur essendo meno frequente con sliding-window su
     volume intero. **Decisione dell'utente**: sì, includerlo — costo quasi
     nullo, rete di sicurezza aggiuntiva.
  2. **Estensione dell'export NIfTI**: quanti soggetti del test set
     esportare per ispezione visiva. **Decisione dell'utente**: tutti i 26,
     non un piccolo campione — per avere l'intero set pronto per analisi
     visiva sistematica.
- **`src/eval_3d.py`** (nuovo):
  - `remove_small_components`: porting diretto (nessun adattamento di shape
    necessario, il volume è già nativamente 3D) da
    `master/src/eval_utils.py`, stessa logica (26-connettività via
    `scipy.ndimage`, soglia configurabile, default 50 voxel, per-classe
    esclusa la classe 0).
  - `predict_volume_3d`: inferenza **nativa** 3D via
    `sliding_window_inference` sul volume intero — **nessun center-crop/
    un-crop richiesto**, a differenza del vecchio `predict_volume` 2D che
    doveva ricostruire il volume impilando slice con crop simmetrico
    (`master/src/eval_utils.py::predict_volume`, §7.1/§7.2 di quel
    progetto). Ritorna sia la predizione post-processata sia quella grezza
    (per confronto dell'effetto del post-processing) sia la ground truth,
    tutte alla risoluzione piena del volume originale.
  - `load_subject_affine`/`save_prediction_nifti`: caricano affine/header
    dal NIfTI di segmentazione originale e li riapplicano alla predizione,
    per un export NIfTI clinicamente valido (apribile in ITK-SNAP/3D
    Slicer con l'orientamento corretto).
  - `evaluate_test_set_3d`: loop completo sul test set — per ciascun
    soggetto, inferenza + post-processing + metriche (via
    `SegmentationMetrics3D`, già pronta dal Punto 4: Dice/HD95 separati per
    ET/NET/CC/ED) + export NIfTI opzionale.
  - `summarize_metrics`: aggrega le metriche per-soggetto in media/std sul
    test set, ignorando esplicitamente i NaN nelle medie HD95 (stessa
    cautela già applicata in `SegmentationMetrics3D.aggregate_and_reset`).
- **Test eseguiti (4/4 passati)**:
  - `remove_small_components`: verificato che una componente isolata di 2
    voxel venga rimossa mentre una componente di 216 voxel sopravviva
    (soglia 50); verificato che classi multiple vengano processate
    indipendentemente senza interferenze reciproche.
  - `get_subject_ids` su dati reali: confermati i 26 soggetti test attesi.
  - **Round-trip NIfTI su un soggetto reale**: caricato l'affine originale,
    esportata una predizione fittizia, ricaricata e verificato che affine e
    dati coincidano esattamente con l'originale.
- **Smoke test end-to-end reale** (`evaluate_test_set_3d` completo, non solo
  unit test): eseguito su 1 soggetto reale del test set (`BraTS-PED-00013-000`,
  copiato in una cartella temporanea), con un modello SegResNet non
  pre-addestrato (verifica puramente strutturale) su CPU:
  - Pipeline completa (caricamento → normalizzazione → sliding-window
    inference roi=32³ → post-processing → metriche → export NIfTI)
    completata senza errori in ~6.5 minuti (CPU; sarà drasticamente più
    rapido su A100).
  - Metriche per-soggetto riportate correttamente **separate per le 4
    sub-regioni** (`dice_ET`, `dice_NET`, `dice_CC`, `dice_ED` e i
    rispettivi `hd95_*`), più `dice_mean_fg`/`hd95_mean_fg`.
  - **Export NIfTI verificato**: file `.nii.gz` riletto con `nibabel`, shape
    `(240, 240, 155)` (risoluzione piena, nessun crop), affine identico a
    quello del BraTS-PEDs originale, etichette nel range corretto
    `{0,1,2,3,4}`.
  - Dati e output temporanei dello smoke test rimossi al termine.
- **Suite di test completa**: 32/32 test passati (4 modelli + 10 loss + 2
  metriche + 3 transform normalizzazione + 9 optim + 4 eval_3d), nessuna
  regressione.

**Cosa NON è stato fatto in questo punto** (rimandato, richiede training
reale su Colab che non è ancora avvenuto): non è stata eseguita una
valutazione reale sui 26 soggetti test con un modello effettivamente
allenato — lo smoke test ha usato un modello non addestrato solo per
verificare che l'intera pipeline di valutazione giri correttamente
end-to-end. L'esecuzione reale su tutti i 26 soggetti con esportazione
NIfTI completa (come richiesto) avverrà dopo il training vero su A100.

---

## Punto 7 — Orchestrazione `run_pipeline_3d.py`: backup + auto-valutazione (roadmap §7)

- **Contesto**: §7 di `road_3D.md` (Orchestrazione) nel progetto originale
  prevedeva un flag `--dim {2d,3d}` dentro `master/run_pipeline.py`. Dato che
  qui si lavora in un workspace `BraTS-PEDs-3D` completamente isolato (per
  scelta esplicita dell'utente all'inizio di questa migrazione),
  quell'orchestrazione è stata reinterpretata come consolidamento di
  `run_pipeline_3d.py` (già creato al Punto 5), che non richiedeva più il
  flag `--dim` (non esiste un ramo 2D in questo progetto) ma mancava ancora
  di alcuni elementi presenti nell'orchestratore 2D originale
  (`master/run_pipeline.py`): backup su cartella esterna ed esecuzione
  automatica della valutazione dopo il training.
- **Tre bivii posti e risolti**:
  1. **Multi-modello**: se estendere `run_pipeline_3d.py` per allenare più
     architetture/loss in sequenza in un solo comando (come `--models unet
     fpn segformer` nel vecchio 2D). **Decisione dell'utente**: no, restare
     a un modello/loss per invocazione — più controllo manuale su Colab,
     checkpoint intermedi più facili da ispezionare tra un run e l'altro.
  2. **Backup checkpoint**: se aggiungere `--backup-dir` per copiare i
     checkpoint su una cartella esterna (es. Google Drive montato),
     essenziale su Colab dove il runtime è effimero. **Decisione
     dell'utente**: sì, aggiungerlo.
  3. **Auto-valutazione**: se invocare automaticamente `src/eval_3d.py` (dal
     Punto 6) subito dopo il training, con un flag `--evaluate/--no-evaluate`.
     **Decisione dell'utente**: sì, con default ON — un solo comando copre
     training+valutazione, come nel vecchio progetto 2D.
- **`run_pipeline_3d.py`** (esteso):
  - Nuovi flag CLI: `--backup-dir`, `--evaluate/--no-evaluate` (default ON),
    `--eval-out-root`, `--eval-min-component-voxels`,
    `--eval-no-postprocessing`.
  - **Tracciamento della history**: il loop di training ora accumula una
    lista di dict per-epoca (`{"epoch", "train", "val"}`) e la salva in
    `<ckpt_dir>/history.json` a fine training (prima non veniva salvata —
    solo stampata a schermo).
  - **`_backup_checkpoints`**: copia `best.pth`/`last.pth`/`history.json` in
    `<backup_dir>/<run_label>/<arch>_<loss>/`, stessa logica di namespacing
    per `run_label` del vecchio `master/run_pipeline.py::_backup` (run
    diversi non si sovrascrivono a vicenda).
  - **`_run_evaluation`**: carica il `best.pth` appena prodotto, invoca
    `evaluate_test_set_3d` (Punto 6) sulla cartella `test/` di `--data-root`,
    con export NIfTI per **tutti** i soggetti del test set (coerente con la
    decisione presa al Punto 6) in
    `evaluation_outputs/<run_label>/<arch>_<loss>/nifti_predictions/`, e
    salva `test_3d_metrics.json` (per-soggetto + summary mean/std). Se
    `--backup-dir` è impostato, copia anche l'intera cartella di
    valutazione (NIfTI inclusi) nel backup.
- **Verifica**: `ast.parse` per la correttezza sintattica, `--help` ispezionato
  per confermare che tutti i nuovi flag siano esposti correttamente.
- **Smoke test end-to-end REALE completo** (non solo unit test): eseguito
  l'intero `run_pipeline_3d.py` su CPU con un mini data-root reale (2
  soggetti train, 1 val, 1 test, copiati da `data/processed_3d/`), con
  `--run-name smoketest7 --backup-dir ... --evaluate` tutti attivi
  contemporaneamente:
  - Training (1 epoca, arch=fpn, loss=dice_focal) → `history.json` salvato
    correttamente.
  - **Backup checkpoint verificato**: `best.pth`/`last.pth`/`history.json`
    presenti sia nella cartella checkpoint originale sia in quella di
    backup.
  - **Valutazione automatica verificata**: eseguita subito dopo il training
    senza intervento manuale, producendo `test_3d_metrics.json` con
    struttura corretta (`dice_ET`/`dice_NET`/`dice_CC`/`dice_ED` e i
    rispettivi `hd95_*` per il soggetto test, più i summary
    `dice_mean_fg_mean`/`std`, `hd95_mean_fg_mean`/`std`) e il NIfTI di
    predizione esportato in `nifti_predictions/`.
  - **Backup della valutazione verificato**: l'intera cartella
    `evaluation_outputs/smoketest7/fpn_dice_focal/` (inclusi i NIfTI) è
    stata copiata correttamente anche nella cartella di backup, in
    `<backup_dir>/evaluation_outputs/smoketest7/fpn_dice_focal/`.
  - Tutti i file/cartelle temporanei dello smoke test rimossi al termine
    (nessun residuo in `data/` o `evaluation_outputs/`).
- **Suite di test completa**: rieseguita come controllo di regressione dopo
  le modifiche a `run_pipeline_3d.py` — 32/32 test ancora passati, invariati
  (le modifiche di questo punto hanno toccato solo l'orchestratore CLI, non
  i moduli `src/` testati unitariamente).

**Cosa NON è stato fatto in questo punto**: nessun training/valutazione
reale su A100/Colab con i 26 soggetti test veri e un modello effettivamente
allenato — resta, come nei punti precedenti, da eseguire quando si passerà
all'infrastruttura cloud.

---

## Punto 8 — `requirements.txt` (roadmap §8)

- **Contesto**: fino a questo punto le dipendenze erano state installate
  incrementalmente nel venv locale (`.venv/`) man mano che servivano ai vari
  moduli, senza un file di requirements dedicato a questo progetto isolato.
- **Due bivii posti e risolti**:
  1. **Struttura del file**: un unico `requirements.txt` con sezioni
     commentate (locale vs Colab) oppure due file separati.
     **Decisione dell'utente**: un unico file con blocchi commentati
     `# --- Fase LOCALE ---` / `# --- Fase COLAB/A100 ---`, per restare
     installabile con un solo comando in entrambi gli ambienti pur
     mantenendo chiara la distinzione Locale/Cloud.
  2. **`huggingface_hub`**: includerlo già ora, pur non essendo importato da
     nessun modulo `src/` esistente (serve solo se si sceglierà la fonte
     BrainSegFounder per i pesi SwinUNETR, decisione lasciata aperta al
     Punto 3). **Decisione dell'utente**: includerlo comunque — libreria
     leggera, evita di dover rieditare il file più avanti.
- **`requirements.txt`** (nuovo): due blocchi commentati (locale:
  `nibabel`/`numpy`/`scipy`/`scikit-learn`/`tqdm`; Colab:
  `torch`/`monai`/`einops`/`huggingface_hub`) più `pytest` condiviso.
  **Nessun pin esatto su `torch`**: solo versione minima, perché su Colab si
  installerà la build con supporto CUDA (diversa dalla build CPU-only
  usata qui in locale per gli smoke test) — pinnare la build locale
  sarebbe stato fuorviante.
- **Verifica REALE di installabilità** (non solo lettura del file): creato un
  venv completamente nuovo e pulito (`/tmp/req_check_venv`), installato
  `pip install -r requirements.txt` da zero — nessun conflitto di
  dipendenze, nessun errore.
  - Verificati tutti gli import chiave (`nibabel`, `numpy`, `scipy`,
    `sklearn`, `tqdm`, `torch`, `monai`, `einops`, `huggingface_hub`,
    `pytest`, incluse le classi `SegResNet`/`SwinUNETR`/`DynUNet` usate da
    `src/models3d.py`) — tutti funzionanti.
  - **Verifica decisiva**: l'intera suite di test del progetto (32 test) è
    stata rieseguita usando **esclusivamente** i pacchetti installati da
    questo `requirements.txt` (nessun pacchetto extra presente nel venv di
    verifica) — **32/32 passati**, a conferma che il file è completo e
    sufficiente, non solo sintatticamente corretto.
  - Venv temporaneo di verifica rimosso al termine.

**Cosa NON è stato fatto in questo punto**: nessun download effettivo dei
pesi pre-addestrati (bundle MONAI `brats_mri_segmentation`, `model_swinvit.pt`
SSL NVIDIA, o BrainSegFounder) — la scelta della fonte per SwinUNETR resta,
come deciso al Punto 3, rimandata al momento del training reale su Colab;
questo punto si è limitato a garantire che le dipendenze *software* necessarie
per caricarli (`monai.bundle`, `huggingface_hub`) siano già elencate.

---

## Punto 9 — Test: `test_dataset_3d.py` (roadmap §9)

- **Contesto**: `road_3D.md` §9 richiede esplicitamente
  `test_dataset3d.py: shape delle patch dopo RandCropByPosNegLabeld`.
  `src/dataset_3d.py` era stato verificato finora solo con script manuali
  inline (Punto 5) — mai con una suite pytest formale e permanente. La
  roadmap nota anche che i vecchi test 2D in `master/tests/` andrebbero
  aggiornati a 5 classi.
- **Due bivii posti e risolti**:
  1. **Ampiezza di `test_dataset_3d.py`**: formalizzare solo ciò già
     verificato a mano al Punto 5, oppure ampliare con casi limite nuovi.
     **Decisione dell'utente**: copertura completa — shape a più ROI,
     augmentation, DTM, edge case (split vuoto, soggetto con file
     mancanti), integrità delle label dopo le trasformazioni geometriche.
  2. **Test 2D in `master/`**: se aggiornarli a 5 classi ora.
     **Decisione dell'utente**: no, resta fuori scope — il workspace
     `BraTS-PEDs-3D` è isolato per regola di ingaggio esplicita; un
     eventuale fix del progetto 2D originale sarà un intervento a parte,
     da richiedere esplicitamente in futuro.
- **`tests/test_dataset_3d.py`** (nuovo, 10 test, tutti su dati reali —
  nessuno skippato in questo ambiente):
  - `build_subject_dicts`: split train reale (205 soggetti attesi), split
    vuoto (lista vuota, nessun errore), soggetto con file NIfTI mancanti
    (dict comunque costruito per convenzione — la responsabilità di
    fallire è di `LoadImaged` a runtime, non della costruzione dei path).
  - **Shape delle patch dopo `RandCropByPosNegLabeld`** (richiesta esplicita
    della roadmap): verificata a **due ROI diverse** (64³ e 128³), non solo
    quella di default.
  - `num_samples` moltiplica correttamente la dimensione del batch (N patch
    per volume → batch di N item con un solo soggetto in input).
  - **Integrità delle label dopo augmentation geometriche** (flip/rotate):
    verificato su 3 soggetti reali con `num_samples=2` che le etichette
    nelle patch restino sempre in `{0,...,4}` — nessuna fuoriuscita dal
    range per interpolazione o bug di trasformazione.
  - Ramo `with_dtm=True`: shape della `distance_map` corretta.
  - `build_eval_transforms`: verificato che il volume di validazione NON
    subisca alcun patch sampling — shape spaziale piena `(240,240,155)`,
    non la ROI di training.
  - `build_dataloaders_3d` end-to-end: batch di train e validazione con le
    shape attese (formalizza permanentemente la verifica manuale già fatta
    al Punto 5).
- **Risultato**: 10/10 test passati (nessuno skippato — tutti eseguiti su
  dati reali in `data/processed_3d/`).
- **Suite di test completa del progetto**: rieseguita per intero come
  controllo di regressione — **42/42 test passati** (32 precedenti + 10
  nuovi), nessuna rottura.

**Cosa NON è stato fatto in questo punto** (deliberatamente, per decisione
dell'utente): nessun aggiornamento dei test 2D in `master/tests/` (che
assumono ancora 4 classi) — resta fuori dal perimetro di questo workspace
isolato, da affrontare come richiesta separata se necessario in futuro.

---

## Punto 2 (notebook) — `01_EDA_3d.ipynb` + `02_preprocessing_3d.ipynb` (roadmap §2, rifatti in forma notebook)

- **Richiesta dell'utente**: riscrivere il Punto 2 (già completato via script
  in precedenza) in forma di **notebook**, simmetrici ai due notebook
  originali del progetto 2D (`master/notebooks/01_EDA.ipynb`,
  `02_preprocessing.ipynb`), con architettura mantenuta identica nella
  struttura a sezioni.
- **Due bivii posti e risolti**:
  1. **Sovrapposizione con gli script esistenti**: `scripts/generate_split.py`
     e `scripts/reorganize_volumes.py` fanno già la stessa cosa.
     **Decisione dell'utente**: i notebook diventano la versione ufficiale;
     gli script restano nel repo come riferimento/utility CLI alternative.
  2. **Formula di stratificazione**: confermata la formula composita già
     validata (`combined_label = et_presence*4 + quartile`, singolo
     `train_test_split` stratificato), non una versione semplificata.
- **`notebooks/01_EDA_3d.ipynb`** (nuovo, 20 celle): stessa struttura a
  sezioni del vecchio EDA 2D (paths → load subject → affine/voxel spacing →
  visualizzazione → intensity distributions → label frequency → bar chart →
  summary), con le correzioni richieste:
  - Legge da `data/raw/`, **5 classi native, nessuna remap** `4→3`.
  - Visualizzazione: preleva la slice con l'area tumorale massima lungo Z
    (calcolata esplicitamente, non solo quella centrale) per le 4 modalità +
    maschera, con legenda e colori aggiornati alla convenzione ET/NET/CC/ED.
  - **Eseguito end-to-end su tutti i 257 soggetti reali** (`jupyter nbconvert
    --execute --inplace`): nessun errore. Output confermati: label native
    `[0,1,2,3,4]` su tutti i soggetti, frequenze per classe coerenti
    (background=99.401%, ET=0.066%, NET=0.441%, CC=0.029% — la più rara, come
    atteso essendo esclusiva pediatrica —, ED=0.063%).
- **`notebooks/02_preprocessing_3d.ipynb`** (nuovo, 20 celle): stessa
  struttura a sezioni del vecchio preprocessing 2D (missing values → demo
  normalizzazione → tumour stats → stratificazione → verifica qualità split →
  export → **estrazione sostituita con Copia Fisica** → verifica integrità →
  summary), con le correzioni richieste:
  - Nessun controllo di outlier/NaN sulle intensità (quello è delegato a
    Colab); qui si verifica solo la presenza dei 5 file NIfTI per soggetto.
  - Cella di normalizzazione mantenuta come **demo visiva non salvata su
    disco** (per continuità didattica con la struttura del vecchio notebook),
    esplicitamente etichettata come tale.
  - Tumour Burden ed ET-presence calcolati sulle **5 classi native** (label 1
    pura per ET), non sulla vecchia label 3 fusa.
  - Stratificazione: **esattamente** la formula composita validata
    (`combined_label = et_presence*4 + quartile`, un solo
    `train_test_split` per train/tmp poi uno per val/test, seed=42).
  - **NIENTE ESTRAZIONE 2D**: la cella di estrazione slice `.npy` del vecchio
    notebook è sostituita da `copy_subject()` — Copia Fisica dei 5 NIfTI
    grezzi per soggetto via `shutil.copy2`, mantenendo le cartelle paziente,
    nessuna modifica ai dati.
- **Verifica rigorosa in due fasi, con due bug trovati e corretti durante il
  test**:
  1. **Bug di setup (mio, non del notebook)**: la prima esecuzione di test è
     stata lanciata da una working directory sbagliata (fuori da
     `notebooks/`), causando un falso `ModuleNotFoundError: No module named
     'src'` — il notebook usa un import relativo (`sys.path.insert(0,
     os.path.abspath('..'))`) che assume di essere eseguito con cwd dentro
     `notebooks/`. Corretto il metodo di test (eseguendo da dentro
     `notebooks/`), non il notebook stesso, che era corretto.
  2. **Bug reale nel notebook**: `ax.boxplot(..., labels=[...])` — il
     parametro `labels` è stato rimosso da matplotlib 3.11 (rinominato in
     `tick_labels`). Corretto con `NotebookEdit` direttamente nel notebook
     originale: `ax.boxplot(..., tick_labels=[...])`.
- **Esecuzione DEFINITIVA completata con successo** (dopo la correzione),
  su richiesta esplicita dell'utente di ripartire puliti:
  - Rimossi preventivamente `data/split_3d.json` e `data/processed_3d/`
    (25GB, dal Punto 2 via script) per garantire un'esecuzione a partire da
    zero, come richiesto.
  - `jupyter nbconvert --execute --inplace notebooks/02_preprocessing_3d.ipynb`
    lanciato in background e lasciato terminare per intero senza interruzioni
    (~10 minuti, dominato dalla copia fisica di 257 soggetti × 5 file NIfTI).
  - **Risultati finali, verificati indipendentemente**:
    - Integrità file: 257/257 soggetti con tutti e 5 i NIfTI attesi.
    - ET-presence (5 classi native): 66.5% (171/257) — confermato di nuovo
      il forte scostamento dal 42.0% del vecchio conteggio post-remap.
    - Split: Train 205 / Val 26 / Test 26, ET rate bilanciato (66.3% / 69.2%
      / 65.4%), **nessuna sovrapposizione tra split** (verificato via
      controllo insiemistico indipendente sui subject_id), 257 soggetti
      unici totali.
    - Copia fisica: conteggio file esatto (1025+130+130 = 1285), verificato
      anche con un controllo indipendente post-hoc (`du -sh` → 25GB).
    - Controllo a campione (9 soggetti, 3 per split) nel notebook: nessun
      errore, label nel range atteso.
  - Output finali (autorevoli, non più i vecchi generati dagli script):
    `data/split_3d.json`, `data/processed_3d/{train,val,test}/`.

**Cosa NON è stato fatto in questo punto**: nessuna modifica a
`scripts/generate_split.py`/`reorganize_volumes.py` (lasciati come
riferimento, non più il percorso "ufficiale" per rigenerare split/dati).

---
