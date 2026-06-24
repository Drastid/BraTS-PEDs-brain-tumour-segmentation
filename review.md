# Code Review — BraTS-PEDs Brain Tumour Segmentation

**Data:** 2026-06-24
**Scope:** Revisione completa di `src/` (9 moduli), `tests/` (3 file), notebook di training/preprocessing, configurazione di progetto.
**Obiettivo della review:** valutare struttura, data loading, definizione modello e training loop *prima* della migrazione su Colab/A100 e dell'implementazione della nuova loss. Niente è stato modificato: questa è una fotografia dello stato attuale.

---

## 1. Sintesi esecutiva

La pipeline è **matura, coerente e ben documentata**. È un progetto 2D slice-based pulito: preprocessing offline → `.npy` per slice → tre architetture drop-in compatibili (U-Net, FPN, SegFormer-B1) → training two-phase identico → valutazione 3D ricostruita con metriche cliniche (Dice/IoU/HD95). I docstring sono di livello professionale e le scelte metodologiche (z-score sui soli voxel cerebrali, augmentation geometrica-only, gestione degli edge-case HD95) sono corrette e ben motivate.

**Giudizio complessivo:** codice solido, pronto per essere scalato. Esistono però alcuni **bug reali e incoerenze** che vanno sistemati *prima* del train-once su Colab — perché entrano nel grafo di training o falsano la riproducibilità, e scoprirli dopo costerebbe un ri-addestramento.

Legenda priorità:
- 🔴 **Alta** — bug o incoerenza che impatta correttezza/riproducibilità/risultati. Da correggere prima del training.
- 🟡 **Media** — debito tecnico o fragilità che conviene sistemare nel contesto Colab.
- 🟢 **Bassa** — pulizia/cosmetica, nessun impatto funzionale.

---

## 2. Struttura del progetto

**Punti di forza**
- Separazione netta `src/` (logica riusabile) vs `notebooks/` (orchestrazione). Gli script CLI (`evaluate_3d_test.py`, `export_nifti_predictions.py`) sono eseguibili come moduli (`python -m src...`) con `PROJECT_ROOT` gestito correttamente.
- `constants.py` centralizza `NUM_CLASSES`, `CROP_SIZE`, `ORIG_SIZE`, `N_SLICES`, `CLASS_NAMES`, `VOXEL_FREQ`: ottima scelta per la coerenza tra moduli.
- Test presenti e mirati sui punti critici (loss, preprocessing voxel-level, metriche 3D + edge-case HD95).

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 2.1 | 🔴 | Configurazione duplicata nei 3 notebook (`03`, `05`, `06`, celle di config) | I blocchi `BATCH_SIZE`, `NUM_WORKERS`, `LR_*`, `*_EPOCHS`, `CROP_SIZE`, pesi loss sono **copia-incollati identici** in tre notebook. Cambiare un iperparametro richiede 3 edit manuali coerenti → fonte di drift. Per il train-once su Colab questo è il rischio numero uno (`implementation_plan.md` STEP 4.3 lo prevede già). **Raccomandazione:** centralizzare in un `src/config.py` (dataclass o dict) importato dai notebook. *(Questa modifica è propedeutica alla Fase 2.)* |
| 2.2 | 🟡 | `constants.py:29` (`CROP_SIZE=192`) vs firme funzioni | `CROP_SIZE` esiste in `constants.py` ma `center_crop()`, `train_one_epoch()`, `evaluate()` hanno `crop_size=192` **hard-coded come default** invece di importare la costante. I notebook ridefiniscono `CROP_SIZE=192` localmente. Tre fonti di verità per lo stesso numero. |
| 2.3 | 🟢 | `src/__pycache__/paths.cpython-314.pyc`, `repro.cpython-314.pyc` | Esistono `.pyc` orfani per moduli `paths.py` e `repro.py` che **non esistono più** come sorgenti. Sono artefatti stale di una versione precedente. Innocui ma fuorvianti (un lettore potrebbe cercarli). Eliminare i `__pycache__` o aggiungere `.gitignore`. |
| 2.4 | 🟢 | `evaluate_3d_test.py:9` | Docstring fa riferimento a `project_review.md` ("action item 5.1"), file non presente nella working dir. Riferimento penzolante. |
| 2.5 | 🟢 | Root del progetto | Cartelle di output versionate (`EDA_01_outputs/`, `training_outputs/`, `evaluation_outputs/`, `comparison_outputs/`) ma nessun `.gitignore` presente nella working dir per `processed_dataset/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`. Da formalizzare prima del push del repo Colab-ready. |

---

## 3. Data loading (`dataset.py`, preprocessing)

**Punti di forza**
- Preprocessing voxel-level corretto e ben motivato: `clip_outliers` (P99.5 sui non-zero) → `zscore_normalise` (statistiche sui soli voxel cerebrali, background forzato a 0). Gestione robusta dei casi degeneri (volume vuoto, σ≈0).
- `BraTSDataset` è leggero: `__init__` fa solo uno scan O(n) della directory, `__getitem__` legge 2 `.npy` → latenza ms. Buon disaccoppiamento I/O (paradigma estrazione-offline / training-veloce).
- Remap label `4→3` gestito sia in `load_subject` sia documentato — coerenza vecchia/nuova convenzione BraTS.

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 3.1 | 🔴 | `dataset.py:240` (`BraTSDataset.__getitem__`, ramo `augment`) | **Bug latente di tipo con albumentations.** Dopo `self.augment(...)`, il codice fa `result["image"].transpose(2,0,1)` e `result["mask"]` assumendo che siano `np.ndarray`. Funziona **solo** perché la pipeline `get_augmentation` non include `ToTensorV2`. Se in futuro si aggiunge `ToTensorV2` (comune), `result["image"]` diventa un `torch.Tensor` e `.transpose(2,0,1)` (semantica NumPy a 3 arg) si rompe. Inoltre il `mask` esce come `int8` e viene convertito a `.long()` a valle: ok, ma fragile. **Raccomandazione:** documentare il contratto in modo esplicito o normalizzare il tipo in uscita. |
| 3.2 | 🔴 | `dataset.py` + `train_utils.center_crop` (interazione preprocessing↔training) | **Augmentation applicata a piena risoluzione 240×240, poi center-crop a 192 nel training loop.** Le slice `.npy` sono salvate 240×240; l'augmentation geometrica (shift/scale/rotate/elastic) gira sul frame intero, e *solo dopo* `train_one_epoch` ritaglia il centro 192×192. Questo è **funzionalmente accettabile** ma comporta: (a) lavoro di augmentation sprecato sui bordi che verranno scartati; (b) `ShiftScaleRotate` con `shift_limit=0.05` può spingere il tumore verso i bordi che poi il crop elimina. Con A100 e crop più ampio (Fase 2) l'effetto cambia: va riconsiderato insieme al nuovo `crop_size`. Segnalato qui perché è un'interazione non ovvia tra due moduli. |
| 3.3 | 🟡 | `dataset.py:223` (`sorted(... os.listdir ...)`) | L'ordinamento dei file è **lessicografico sulle stringhe**. I nomi usano slice zero-padded (`_slice007`), quindi l'ordine è corretto *per come sono generati oggi*. Ma se un dataset cross-domain producesse padding diverso (`_slice7`), l'ordine si romperebbe. Robustezza: ordinare per indice numerico parsato (come già fa `eval_utils._parse_slice_idx`). Incoerenza tra i due moduli sullo stesso problema. |
| 3.4 | 🟡 | `dataset.py:235-236` (commento "mmap-backed") | Il commento dice "mmap-backed disk reads", ma `np.load` senza `mmap_mode='r'` **carica l'intero array in RAM**, non fa mmap. Il commento è fuorviante. O si aggiunge `mmap_mode='r'` (utile con `num_workers` alti su A100) o si corregge il commento. |
| 3.5 | 🟡 | Preprocessing notebook, `extract_split` (cella 19) | Nessun **subsampling delle slice background-only**: vengono salvate *tutte* le slice valide (39.538 totali). Molte slice non contengono tumore → forte sbilanciamento a livello di slice oltre a quello a livello di voxel. Non è un bug, ma è un parametro di design rilevante per il training (e per la Fase 2: con A100 si può permettere di tenerle tutte, ma andrebbe una scelta consapevole). |

---

## 4. Definizione del modello (`models.py`)

**Punti di forza**
- `SegFormerWrapper` è una soluzione elegante: espone `.encoder` e `.decode_head` come property per restare *drop-in compatibile* con il training loop pensato per `segmentation-models-pytorch`. Il `forward` riporta i logit da H/4×W/4 alla risoluzione input via `F.interpolate`. Pulito.
- L'**adattamento patch-embedding 3→4 canali** è fatto correttamente: pesi RGB preservati, 4° canale = media(RGB), bias copiato, stride/padding ereditati. Inizializzazione sensata e ben documentata.
- `get_segformer` usa `ignore_mismatched_sizes=True` per re-inizializzare la testa su 4 classi: corretto.

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 4.1 | 🟡 | `models.py:96` (`F.interpolate` in `forward`) vs training loop | Il wrapper SegFormer fa upsampling **bilineare** dei logit a risoluzione piena *dentro* il forward. Per U-Net/FPN i logit escono già a piena risoluzione. È coerente per la loss, ma significa che SegFormer paga un upsample 4× ad ogni step e il segnale di loss è calcolato su logit interpolati (non nativi). È una scelta legittima ma asimmetrica tra i modelli — da tenere a mente quando si confrontano i risultati. |
| 4.2 | 🟡 | `models.py` (factory) | **Niente factory unificata per i tre modelli.** U-Net e FPN sono istanziati a mano nei notebook con `smp.Unet(...)`/`smp.FPN(...)` (parametri ripetuti), SegFormer ha `get_segformer`. Una `build_model(arch, ...)` unica ridurrebbe la duplicazione e abiliterebbe la selezione-da-config richiesta dall'`implementation_plan.md`. |
| 4.3 | 🟢 | `models.py` | `SegFormerWrapper.forward` usa `align_corners=False` (corretto per segmentazione), ma il valore è hard-coded. Nessun impatto pratico; solo nota di completezza. |

---

## 5. Loss (`losses.py`)

**Punti di forza**
- `DiceLoss` e `FocalLoss` accettano **logit grezzi** (softmax/log-softmax internamente) — contratto chiaro e numericamente stabile (`log_softmax` invece di `log(softmax)`).
- `FocalLoss` implementa correttamente `-α_t·(1-p_t)^γ·log(p_t)` con gather sulle classi vere; il test `test_focal_loss_zero_gamma_equals_ce` verifica la riduzione a CE per γ=0. Buona copertura.
- `DiceLoss` calcola il Dice **per-campione** (`dim=(1,2)`) poi media sul batch: corretto (evita di mescolare i denominatori tra immagini diverse).

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 5.1 | 🔴 | `losses.py:124` + `CombinedLoss.__init__:273` (interazione `class_weights` ↔ `ignore_background`) | **I pesi per-classe del Dice vengono silenziosamente ignorati nella configurazione effettivamente usata.** `CombinedLoss` passa `class_weights` al `DiceLoss` *solo se* `ignore_background=False` (riga 273: `class_weights if not ignore_background else None`). Ma tutti i notebook usano `ignore_background=True` → **il Dice non è mai pesato**; i pesi inverse-frequency agiscono solo sulla Focal (α). Questo è probabilmente *intenzionale* (il commento lo giustifica), ma è una **trappola di interfaccia**: l'utente passa `class_weights` al Dice credendo abbiano effetto, e non ce l'hanno. Inoltre, internamente `DiceLoss.forward` (riga 124) applica i pesi *solo se* `class_weights is not None AND not ignore_background` — doppia condizione che rende i pesi morti nel caso d'uso reale. **Raccomandazione:** o si abilita un weighting foreground-only del Dice, o si documenta esplicitamente che con `ignore_background=True` i pesi Dice sono ignorati e si emette un warning. Rilevante perché la **Fase 3 introdurrà pesi globali `w_k` per la GSL** — bisogna chiarire dove i pesi agiscono davvero. |
| 5.2 | 🟡 | `losses.py:220` (`FocalLoss`, gestione `ignore_index`) | `ignore_index` di default è `-100` (convenzione PyTorch), ma il masking scatta solo se `self.ignore_index >= 0` (riga 220). Con il default `-100` il ramo non si attiva mai e i `clamp(min=0)` su `targets_flat` (righe 204-205) gestiscono valori negativi mappandoli alla classe 0. È coerente con l'uso attuale (nessun void label), ma la logica `ignore_index >= 0` è controintuitiva rispetto alla convenzione `-100`. Documentare o normalizzare. |
| 5.3 | 🟡 | `losses.py:75-77` (normalizzazione pesi Dice) | `DiceLoss` L1-normalizza i `class_weights` (`w / w.sum()`) ma poi nel caso foreground-only fa `self.class_weights[start_c:]` **senza ri-normalizzare** la sotto-selezione → la somma dei pesi foreground non è più 1. Inconsistenza minore (incide solo sulla scala della loss, non sul gradiente relativo), ma vale la pena saperlo, soprattutto in vista della combinazione α(t)·DiceFocal della Fase 3. |
| 5.4 | 🟢 | `losses.py` (`CombinedLoss.forward` return) | Ritorna una tupla `(total, d_loss, f_loss)` invece di un singolo scalare. Comodo per il logging, ma rompe la convenzione `nn.Module` standard (criterion che ritorna scalare). I chiamati lo gestiscono correttamente; segnalato solo perché la Fase 3 dovrà estendere questa firma (per loggare anche la GSL e α(t)) — pianificare l'estensione della tupla/dict ora. |

---

## 6. Training loop (`train_utils.py`, notebook)

**Punti di forza**
- Two-phase schedule ben implementato: freeze encoder (5 ep) → unfreeze con **LR differenziale** (encoder 0.1×, decoder 1×) + `CosineAnnealingLR` per fase. AdamW, weight decay 1e-4, AMP, grad clip 1.0. Metodologia corretta e robusta.
- AMP gestito correttamente: `scaler.scale().backward()` → `unscale_` → `clip_grad_norm_` → `step` → `update`. L'ordine è giusto (unscale prima del clip).
- `MetricTracker` come running-mean pesato per batch size: corretto. `_compute_batch_dice` sotto `no_grad` per il logging.
- Pre-flight sanity check nei notebook (hardware/dataloader/single-step): ottima pratica per non bruciare GPU su bug banali.

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 6.1 | 🔴 | `train_utils.py:299` (`autocast` device_type) vs notebook SegFormer | **Incoerenza AMP tra il modulo e il notebook SegFormer, con rischio di instabilità.** In `train_one_epoch` la loss è calcolata **dentro** `autocast` (riga 299-301). Ma il pre-flight check del notebook SegFormer (cella 6) calcola deliberatamente la loss **fuori** da autocast (`_logits.float()`) con commento "compute outside autocast for stability". Il training reale (che usa `train_one_epoch`) **non** segue questa cautela → la `CombinedLoss` (con softmax/log-softmax + Dice su somme) gira in fp16/bf16. Per la Dice questo è di solito ok, ma il `1e-6` di smoothing e le somme su molti pixel possono soffrire in fp16. **Raccomandazione:** uniformare — calcolare almeno la loss in fp32 anche in `train_one_epoch`, o documentare che ci si affida al GradScaler. Da decidere prima del train-once. |
| 6.2 | 🔴 | `train_utils.py:299` (`str(device).split(":")[0]`) | **Fragilità nel parsing del device per autocast.** `device_type=str(device).split(":")[0]` funziona per `"cuda"`/`"cuda:0"`/`"cpu"`, ma se `device` è un `torch.device` senza indice il `str()` dà `"cuda"` (ok) — tuttavia su CPU `autocast(device_type="cpu", enabled=False)` è inutile overhead. Minore, ma con A100 conviene un handling esplicito. Più importante: **non c'è `bf16`**. L'A100 supporta `bfloat16`, più stabile dell'fp16 per le loss — da valutare in Fase 2 (`dtype=torch.bfloat16` in autocast). |
| 6.3 | 🔴 | Notebook training (celle config) — riproducibilità | **Determinismo incompleto.** I notebook fissano `SEED=42` su `random`/`numpy`/`torch`/`cuda`, ma **non** impostano `torch.backends.cudnn.deterministic=True` / `benchmark=False`, né `torch.use_deterministic_algorithms`, né il `worker_init_fn`/`generator` del DataLoader. Con `num_workers=4` e `shuffle=True` l'ordine e l'augmentation **non sono riproducibili** tra run. L'`implementation_plan.md` (STEP 1.4) richiede esplicitamente determinismo per confrontare le ablazioni — va sistemato prima del train-once, altrimenti i confronti di Fase C non sono validi. |
| 6.4 | 🟡 | `train_utils.py` (`save_checkpoint`/`load_checkpoint`) | Il checkpoint salva `model/optimizer/scheduler/epoch/metrics/scaler` ma **non** lo stato RNG (`torch.get_rng_state`, `numpy`, `random`, `cuda`), né l'eventuale `fold`. Per il resume robusto su Colab (sessioni che cadono, STEP 1.2 del piano) manca il pezzo che garantisce continuità esatta. Inoltre `best_val_dice` vive solo nel notebook → un resume lo perde. |
| 6.5 | 🟡 | Notebook training (loop best checkpoint) | **Nessun early stopping né `last.pth` per-epoca.** `last.pth` è salvato solo a fine fase, non ad ogni epoca. Se Colab si disconnette a metà Fase 2 si perdono fino a 24 epoche di training. Con 30 epoche fisse e nessun early-stop, inoltre, si rischia overfitting non monitorato oltre il best (il README mostra best epoch 14-18 su 30 → le ultime ~12 epoche potrebbero essere sprecate o dannose). |
| 6.6 | 🟡 | `train_utils.py:306,311` (`clip_grad_norm_(model.parameters())`) | Il grad clipping itera su `model.parameters()` (tutti), inclusi i parametri **frozen** in Fase 1 (encoder con `requires_grad=False`). `clip_grad_norm_` ignora i `.grad is None`, quindi è corretto, ma calcola la norma su un insieme che include parametri senza gradiente — innocuo ma leggermente impreciso. Coerenza: clippare solo i parametri con gradiente. |
| 6.7 | 🟢 | `train_utils.py:354` vs argomenti `evaluate` | `evaluate` accetta `crop_size` e applica `center_crop` — corretto e coerente con `predict_volume`. Nessun problema; nota di coerenza positiva. |

---

## 7. Valutazione (`eval_utils.py`, `evaluate_3d_test.py`)

**Punti di forza**
- Gestione degli **edge-case HD95 eccellente e ben documentata**: Caso A (entrambi vuoti)→NaN, Caso B (miss completo)→NaN+warning, Caso C (falso positivo)→NaN+warning, eccezioni medpy catturate. Aggregazione con `np.nanmean` e conteggio NaN separato. Questo è il punto più curato del codice.
- `predict_volume` riapplica **lo stesso center-crop** del training e fa l'**un-crop** prima di assemblare il volume — coerenza spaziale train/inference garantita. `infer_dataset_shape` per auto-detect dimensioni (cross-domain ready).
- `remove_small_components` con 26-connettività per-classe: corretto per eliminare il "confetti effect" dei modelli 2D.
- `compute_dice_volume`/`compute_iou_volume`/`compute_hd95_volume` testati con casi perfetti ed edge-case.

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 7.1 | 🔴 | `eval_utils.py:289-292` (`predict_volume`, un-crop asimmetrico) | **Asimmetria pred vs GT nella regione esterna al crop.** La predizione viene scritta **solo** nella regione centrale 192×192 (`pred_vol[top:top+crop, ...]`), mentre la GT è salvata a **piena 240×240** (`gt_vol[:,:,sl_idx] = msk`). Conseguenza: qualunque voxel di tumore presente nella GT *fuori* dalla fascia centrale 192×192 è conteggiato come **falso negativo automatico** (la pred lì è forzata a 0). Per BraTS-PEDs i tumori sono tipicamente centrali, quindi l'impatto pratico è piccolo, ma **la metrica è sistematicamente penalizzata** in modo non simmetrico rispetto a come è stato fatto il training. Almeno andrebbe documentato/quantificato; idealmente la GT andrebbe confrontata sulla stessa regione, oppure si predice a piena risoluzione. **Diventa più rilevante in Fase 2** se si aumenta `crop_size` (240-full elimina il problema). |
| 7.2 | 🔴 | `eval_utils.py` ↔ `train_utils.center_crop` (duplicazione formula crop) | La formula `top=(orig-crop)//2` è **reimplementata** in `predict_volume` (riga 265), in `center_crop` (train_utils riga 219), e pre-calcolata in `eval_utils` (`_CROP_TOP`, riga 60). Tre copie della stessa logica. Se cambia la strategia di crop (Fase 2), il rischio di disallineamento train/eval è alto → bug silenzioso che falserebbe tutte le metriche. **Unificare in un'unica funzione condivisa.** |
| 7.3 | 🟡 | `evaluate_3d_test.py:61` (`DEVICE` modulo-level) | `DEVICE` è risolto a import-time del modulo. `export_nifti_predictions.py` importa `DEVICE` da `evaluate_3d_test` — accoppiamento che funziona ma rende difficile testare/forzare CPU. Minore. |
| 7.4 | 🟡 | `evaluate_3d_test.py:181-204` (loop subject) vs salvataggio | Lo script calcola e salva **solo le metriche** (JSON per-subject), ma **non le predizioni volumetriche né le probabilità**. L'`implementation_plan.md` (STEP 5.1) segnala questo come il rischio più grande per le analisi post-hoc (ensemble, post-processing ET/WT, incertezza): senza salvare le predizioni servirà ri-eseguire l'inferenza. Non è un bug, ma è un gap architetturale noto da colmare prima/durante il train-once. |
| 7.5 | 🟡 | `eval_utils.py:530` (`from medpy.metric.binary import hd95` dentro la funzione) | Import lazy dentro `compute_hd95_volume` — chiamato in un loop per ogni subject×classe. L'import è cached da Python dopo la prima volta (innocuo), ma stilisticamente l'import andrebbe a livello modulo o fatto una volta. Cosmetico. |
| 7.6 | 🟢 | `evaluate_3d_test.py:381` (`torch.cuda.empty_cache()`) | Chiamato incondizionatamente anche se `DEVICE` è CPU. Innocuo (no-op se CUDA non c'è) ma in `export_nifti_predictions.py` la stessa chiamata è correttamente protetta da `if torch.cuda.is_available()` — incoerenza minore tra i due script. |

---

## 8. Test (`tests/`)

**Punti di forza**
- Test mirati e significativi: Dice perfetto/invertito, Focal==CE per γ=0, clip outlier, z-score, remove_small_components con blob di taglie note, edge-case HD95 completi.
- `pytest.ini` con `pythonpath=.` configurato correttamente.

**Problemi**

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 8.1 | 🟡 | Copertura `train_utils.py` e `models.py` | **Nessun test** per `train_one_epoch`/`evaluate` (anche solo smoke test su un mini-batch sintetico), né per l'adattamento 4-canali di `get_segformer` (verificare che `proj.weight` abbia shape `[C_out,4,k,k]` e che i primi 3 canali siano preservati). Sono i pezzi che la Fase 2/3 toccheranno di più → conviene una rete di sicurezza prima di modificarli. |
| 8.2 | 🟡 | `CombinedLoss` non testata | I test coprono `DiceLoss` e `FocalLoss` separate ma **non** `CombinedLoss` né l'interazione `class_weights`/`ignore_background` (vedi §5.1). Un test che fissasse il comportamento attuale renderebbe sicura la modifica della Fase 3. |
| 8.3 | 🟢 | `predict_volume` non testata | Comprensibile (richiede `.npy` su disco), ma un test con tmpdir e 2-3 slice sintetiche blinderebbe la coerenza crop/un-crop (§7.1, §7.2). |

---

## 9. Sicurezza / robustezza minori

| # | Pri | File / Posizione | Osservazione |
|---|-----|------------------|--------------|
| 9.1 | 🟡 | `train_utils.py:436` (`torch.load(..., weights_only=False)`) | `weights_only=False` esegue unpickle arbitrario. Per checkpoint propri va bene, ma è una pratica che PyTorch sta deprecando come default per sicurezza. Se i checkpoint verranno scaricati da Drive condiviso, valutare `weights_only=True` + salvataggio del solo `state_dict` dove possibile. |
| 9.2 | 🟢 | `eval_utils.py` / `evaluate_3d_test.py` | `warnings.warn(..., RuntimeWarning)` usato per gli edge-case HD95: corretto, ma in un loop su 26 subject × 3 classi può generare molto rumore. La cattura con `catch_warnings(record=True)` nello script è la scelta giusta — solo nota. |

---

## 10. Conclusioni e priorità d'azione (pre-Colab)

Il codice è **pronto a essere scalato**, ma queste voci 🔴 vanno affrontate *prima* del train-once su Colab, perché toccano correttezza, riproducibilità o entrano nel grafo di training:

1. **§6.3 — Determinismo completo** (cudnn deterministic, worker seeding). Senza, i confronti tra modelli/ablazioni non sono validi.
2. **§5.1 — Chiarire dove agiscono i `class_weights`** (Dice ignorato con `ignore_background=True`). Propedeutico alla loss di Fase 3.
3. **§6.1 / §6.2 — Stabilità AMP** (loss in fp32, valutare bf16 su A100). Da decidere prima di addestrare.
4. **§7.1 / §7.2 — Asimmetria pred/GT nell'un-crop e formula crop duplicata.** Unificare il crop in un'unica funzione; rivalutare con il nuovo `crop_size` della Fase 2.
5. **§2.1 — Centralizzare la config** (un `config.py`). Abilita Fase 2 e train-once senza editare 3 notebook.
6. **§3.1 / §3.3 / §3.4 — Robustezza data loading** (contratto augmentation, ordinamento numerico, commento mmap).

Voci 🟡 (checkpoint RNG-state, last.pth per-epoca, factory modello unificata, salvataggio predizioni, test su train/model/loss) sono fortemente consigliate nel contesto Colab ma non bloccanti per la correttezza del singolo run.

I 🟢 sono pulizia opportunistica.

> **Nota di scope:** questa review fotografa lo stato attuale. La **Fase 2** (scaling A100) interverrà su batch/crop/workers/profondità e su diversi punti qui segnalati (§3.2, §6.2, §7.1); la **Fase 3** (loss α(t)·DiceFocal + (1-α(t))·GSL con pesi globali) interverrà su `losses.py` e sul training loop, dove i punti §5.1, §5.3, §5.4 e §6.4 sono direttamente rilevanti.
