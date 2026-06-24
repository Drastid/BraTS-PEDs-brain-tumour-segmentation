# Modifiche di Scaling per A100 — Fase 2

**Data:** 2026-06-24
**Target hardware:** Google Colab — NVIDIA A100 (40 GB; le indicazioni valgono anche per la variante 80 GB).
**Origine:** in locale la pipeline era vincolata da **8 GB di VRAM** (RTX 3080 Laptop). Questo documento identifica i parametri che fungevano da collo di bottiglia e propone i nuovi valori per saturare l'A100 **senza stravolgere la pipeline 2D slice-based** (stessa architettura, stesso paradigma estrazione-offline → training, stessa loss e schedule two-phase).

> **Prerequisito già soddisfatto (Blocco 1):** dopo il fix §2.1, quasi tutti gli iperparametri vivono in **un unico punto**, [src/config.py](src/config.py). La maggior parte delle modifiche qui sotto si applica quindi *una sola volta* in `config.py`, non più in 3 notebook. Dove serve un intervento altrove (notebook, `constants.py`, script di eval) è indicato esplicitamente.

---

## 0. Come leggere questo documento

Ogni voce riporta: **parametro · file:riga (o funzione) · valore attuale → valore A100 proposto · perché era un collo di bottiglia · cautele**.

Priorità di impatto sulle prestazioni / utilizzo GPU:
- 🟥 **Alto impatto** — sblocca direttamente throughput o qualità (batch, crop, modello, bf16).
- 🟧 **Medio** — migliora l'efficienza della pipeline dati (workers, prefetch).
- 🟦 **Tuning fine** — flag e accorgimenti che spremono l'ultimo margine.

---

## 1. 🟥 Batch size — il collo di bottiglia primario

| | |
|---|---|
| **Parametro** | `batch_size` |
| **File · riga** | [src/config.py:66](src/config.py#L66) — `batch_size: int = 16` |
| **Valore attuale** | `16` |
| **Proposto A100 (40GB)** | **64** (con `crop=192`/`224`); **48** se si passa a `crop=240` + encoder più profondo |
| **Proposto A100 (80GB)** | **96–128** |

**Perché era un collo di bottiglia:** con 8 GB, batch=16 a 192×192 su ResNet34 era vicino al limite di memoria. L'A100 ha 40/80 GB → si può quadruplicare/ottuplicare il batch, aumentando il throughput (meno step per epoca) e stabilizzando le statistiche del gradiente sulle classi rare (NCR/ET) — utile dato lo sbilanciamento estremo.

**⚠️ Cautela obbligatoria — ricalibrare il learning rate.** Aumentare il batch senza scalare l'LR degrada la convergenza. Regola pratica: **square-root scaling** (più conservativo del linear per fine-tuning di reti pretrained).
- Da batch 16 → 64 (×4): LR × √4 = **×2**.
- Quindi: `lr_phase1` 3e-4 → **6e-4**, `lr_phase2` 1e-4 → **2e-4** ([src/config.py:74-75](src/config.py#L74-L75)).
- In alternativa, mantenere gli LR attuali e usare un **warmup lineare** di 1-2 epoche (più sicuro; vedi §7).

**Cautela secondaria:** `BatchNorm` di ResNet beneficia di batch grandi, ma con batch molto alto valutare che il `drop_last=True` (già presente nei notebook) non scarti troppi campioni — irrilevante con 32k+ slice di training.

---

## 2. 🟥 Crop size — patch spaziale ristretta dalla memoria

| | |
|---|---|
| **Parametro** | `crop_size` (e la costante sorgente `CROP_SIZE`) |
| **File · riga** | [src/config.py:63](src/config.py#L63) — `crop_size: int = CROP_SIZE`; valore base in [src/constants.py:29](src/constants.py#L29) — `CROP_SIZE: int = 192` |
| **Valore attuale** | `192` (su slice originali `ORIG_SIZE=240`) |
| **Proposto A100** | **240 (full slice)** — preferito; in alternativa **224** |

**Perché era un collo di bottiglia:** il center-crop a 192×192 (da 240×240) era un compromesso per ridurre l'attivazione in memoria su 8 GB. Costava la perdita di una corona di 24 px per lato.

**Vantaggio specifico su A100 — chiude un bug noto.** Portare `crop_size` a **240** elimina del tutto l'asimmetria pred/GT discussa nel fix **§7.1**: a piena risoluzione non c'è più regione esclusa, quindi nessun voxel di tumore periferico viene scartato. È l'opzione più pulita: più contesto spaziale **e** metrica non penalizzata.

**Come applicarlo (single source of truth, grazie a §7.2):**
- Opzione consigliata: impostare `crop_size=240` in `config.py`. La funzione condivisa `compute_crop_offsets(240, 240)` ritorna `(0,0)` → nessun crop effettivo, gestito correttamente sia in training (`center_crop`) sia in inferenza (`predict_volume`), senza modifiche al codice.
- ⚠️ Se si sceglie 224 (non 240): 224 non è divisibile per 32. ResNet/SMP gestiscono dimensioni arbitrarie via padding interno, ma **SegFormer** (patch 4×4, downsampling ×32) è più sensibile — 224 è divisibile per 32 (224 = 7×32) quindi è sicuro; **evitare** valori non multipli di 32 per SegFormer.

**⚠️ Interazione con l'augmentation (§3.2 della review):** oggi l'augmentation gira a 240 e poi si croppa a 192. Passando a crop=240 l'augmentation geometrica copre l'intera slice usata → comportamento più naturale; nessuna modifica al codice necessaria, ma è un cambiamento di regime da tenere a mente nel confronto con i risultati 8GB.

**Costo memoria:** da 192² a 240² è ×1.56 di area → combinare con il batch (es. crop=240 + batch=48 invece di 64). Vedi tabella budget §8.

---

## 3. 🟥 Profondità / capacità del modello — encoder e hidden dims

Questa è la leva di **qualità** più importante, e con 8 GB non era pienamente esplorabile. Resta **drop-in** (la pipeline non cambia: stessi `[B,4,H,W]` → `[B,4,H,W]`).

### 3a. U-Net / FPN — encoder backbone

| | |
|---|---|
| **Parametro** | `encoder` |
| **File · riga** | [src/config.py:58](src/config.py#L58) — `encoder: str = "resnet34"` |
| **Valore attuale** | `resnet34` (~24.4M U-Net / ~22.1M FPN) |
| **Proposto A100** | **`resnet50`** (bilanciato) o **`se_resnext50_32x4d`** (qualità superiore) |

**Perché era limitato:** ResNet34 era una scelta di compromesso memoria/velocità per 8 GB. Su A100 si può salire a ResNet50 (più capacità rappresentativa, feature più ricche per i bordi tumorali → potenziale guadagno su HD95) restando ampiamente nel budget.

**⚠️ Da aggiornare in DUE punti** (l'encoder è usato anche fuori dai notebook):
1. [src/config.py:58](src/config.py#L58) — per il training.
2. [src/evaluate_3d_test.py:72](src/evaluate_3d_test.py#L72) — `ENCODER = "resnet34"` per il caricamento in valutazione 3D. **Deve coincidere** con quello di training, altrimenti il checkpoint non si carica. *(Nota: in un secondo momento converrebbe far leggere a `evaluate_3d_test.py` lo stesso `config.py`, ma per ora è un valore da allineare a mano.)*

### 3b. SegFormer — variante del backbone (hidden dims / profondità)

| | |
|---|---|
| **Parametro** | `segformer_checkpoint` |
| **File · riga** | [src/config.py:60](src/config.py#L60) — `segformer_checkpoint: str = "nvidia/mit-b1"` |
| **Valore attuale** | `nvidia/mit-b1` (~13.7M) |
| **Proposto A100** | **`nvidia/mit-b3`** (~47M) o **`nvidia/mit-b2`** (~27M, più conservativo) |

**Perché era limitato:** B1 è il più piccolo della famiglia MiT; scelto per 8 GB. Le varianti B2/B3 hanno hidden dimensions e profondità maggiori (più stadi/canali nel transformer gerarchico) → maggiore capacità, storicamente forti su BraTS.

**Vantaggio:** SegFormer era già il modello migliore (vedi README); aumentarne la capacità su A100 è l'investimento a più alto ritorno atteso.

**⚠️ Cautele:**
- Il fix 4-canali in [src/models.py:107](src/models.py#L107) (`get_segformer`) è **agnostico alla variante**: opera su `patch_embeddings[0].proj` qualunque sia B1/B2/B3. Nessuna modifica al codice necessaria — basta cambiare la stringa in config.
- Allineare anche il caricamento in valutazione: [src/evaluate_3d_test.py:131-136](src/evaluate_3d_test.py#L131-L136) (`load_segformer_model` chiama `get_segformer("nvidia/mit-b1", ...)` — passare la stessa variante).
- Modelli più grandi → ridurre leggermente il batch (B3 + crop 240 → batch 32–48).

---

## 4. 🟧 Numero di worker del DataLoader

| | |
|---|---|
| **Parametro** | `num_workers` |
| **File · riga** | [src/config.py:67](src/config.py#L67) — `num_workers: int = 4` |
| **Valore attuale** | `4` |
| **Proposto A100** | **8** (Colab A100 espone ~12 vCPU) |

**Perché era un (mezzo) collo di bottiglia:** 4 worker erano prudenti per il locale (ed evitavano problemi multiprocessing su Windows — il notebook di preprocessing nota `num_workers=0` "Windows safe"). Su Colab (Linux) con A100 e batch più grandi, il caricamento `.npy` dal disco può diventare il fattore limitante se la GPU resta affamata. 8 worker alimentano meglio una A100.

**⚠️ Cautele:**
- Non superare il numero di vCPU; oltre ~8 il ritorno è marginale e aumenta la RAM host.
- Il **seeding deterministico dei worker** (fix §6.3, `worker_init_fn=seed_worker` + `generator`) resta valido e necessario con più worker — già applicato nei notebook nel Blocco 1.
- Combinare con i parametri di §6 (`persistent_workers`, `prefetch_factor`) per il massimo beneficio.

---

## 5. 🟥 Precisione mista: fp16 → bf16

| | |
|---|---|
| **Parametro** | `amp_dtype` |
| **File · riga** | [src/config.py:91](src/config.py#L91) — `amp_dtype: str = "fp16"` |
| **Valore attuale** | `fp16` (comportamento legacy con GradScaler) |
| **Proposto A100** | **`bf16`** |

**Perché ora è possibile (e migliore):** l'A100 (Ampere) ha supporto hardware nativo per **bfloat16**, che ha lo stesso range di esponente dell'fp32 → niente problemi di overflow/underflow e **nessun bisogno di loss scaling**. Più stabile dell'fp16 per le riduzioni di Dice/Focal (già reso fp32-safe nel fix §6.1).

**Come applicarlo (infrastruttura già pronta dal Blocco 1):**
- Impostare `amp_dtype="bf16"` in config.
- `train_one_epoch` già gestisce bf16: con `amp_dtype="bf16"` salta automaticamente il GradScaler (vedi `use_scaler` in [src/train_utils.py](src/train_utils.py)). Nel notebook si può quindi passare `scaler=None` o lasciare lo scaler disabilitato — il codice lo ignora correttamente per bf16.
- ⚠️ **Azione richiesta nel notebook (loop di training):** poiché nel Blocco 1 (approccio minimale) le celle del loop non sono state toccate, le chiamate `train_one_epoch(..., scaler=scaler)` usano il default `amp_dtype="fp16"`. Per attivare bf16 occorre **passare `amp_dtype=AMP_DTYPE`** a `train_one_epoch` nelle celle del loop (Phase 1 e Phase 2) dei tre notebook. È una modifica di una riga per chiamata.

---

## 6. 🟧 Efficienza del DataLoader (parametri secondari)

Da aggiungere nelle celle DataLoader dei notebook (oggi non presenti). Migliorano l'alimentazione della GPU senza rischi.

| Parametro | Dove | Attuale | Proposto | Effetto |
|---|---|---|---|---|
| `persistent_workers` | DataLoader (notebook) | assente (`False`) | **`True`** | Evita di ricreare i worker ad ogni epoca → meno overhead, importante con 30 epoche |
| `prefetch_factor` | DataLoader (notebook) | default (2) | **4** | Più batch pre-caricati per worker → GPU meno in attesa |
| `pin_memory` | [src/config.py:68](src/config.py#L68) | `True` | `True` (invariato) | Già ottimale per transfer host→GPU |

**Esempio (train_loader):** aggiungere `persistent_workers=True, prefetch_factor=4` alla chiamata `DataLoader(...)`. ⚠️ `persistent_workers=True` richiede `num_workers > 0` (soddisfatto).

---

## 7. 🟦 Schedule e durata del training (opzionale, abilitato da A100)

Non sono colli di bottiglia di memoria, ma l'A100 rende economico migliorarli.

| Parametro | File · riga | Attuale | Proposto | Note |
|---|---|---|---|---|
| `phase2_epochs` | [src/config.py:72](src/config.py#L72) | `25` | `25` → valutare **30-40** | Il README mostra best epoch 14-18; con LR ricalibrato e modello più grande il punto ottimo può spostarsi. Più epoche = più costo, ma su A100 è sostenibile. **Solo se** si aggiunge early stopping (vedi review §6.5), altrimenti rischio overfitting. |
| Warmup LR | loop training (notebook) | assente | **1-2 epoche lineari** | Alternativa/complemento al re-scaling LR di §1: stabilizza l'avvio con batch grande. Implementabile con `torch.optim.lr_scheduler.LinearLR` + `SequentialLR` davanti al `CosineAnnealingLR` esistente. |

---

## 8. Budget memoria indicativo e configurazioni consigliate

Stime di massima per A100 40 GB con AMP bf16 (l'occupazione reale va verificata con `nvidia-smi` / `torch.cuda.max_memory_allocated()` dopo il primo step — il sanity-check già presente nei notebook stampa la VRAM).

| Configurazione | Modello | crop | batch | VRAM stimata | Quando usarla |
|---|---|---|---|---|---|
| **A — Conservativa** | ResNet34 / B1 (invariati) | 240 | 48 | ~12-16 GB | Primo run di smoke test su A100, baseline confrontabile |
| **B — Bilanciata (consigliata)** | ResNet50 / SegFormer-B2 | 240 | 48-64 | ~22-28 GB | Miglior rapporto qualità/sicurezza |
| **C — Massima capacità** | SE-ResNeXt50 / SegFormer-B3 | 240 | 32-48 | ~32-38 GB | Spremere l'A100 40GB; verificare di non andare OOM |

> Per A100 **80 GB**: raddoppiare i batch della rispettiva riga.

---

## 9. Riepilogo operativo — checklist modifiche

**In [src/config.py](src/config.py) (un solo punto, grazie al fix §2.1):**
1. ☐ `batch_size`: 16 → **64** (riga 66) — *config B*
2. ☐ `crop_size`: tenere il default ma impostare a **240** (riga 63 / o `CROP_SIZE` in constants.py riga 29)
3. ☐ `encoder`: `resnet34` → **`resnet50`** (riga 58)
4. ☐ `segformer_checkpoint`: `mit-b1` → **`mit-b2`** (riga 60)
5. ☐ `num_workers`: 4 → **8** (riga 67)
6. ☐ `amp_dtype`: `fp16` → **`bf16`** (riga 91)
7. ☐ `lr_phase1`/`lr_phase2`: ricalibrare per il batch (riga 74-75) — **6e-4 / 2e-4** con sqrt-scaling, oppure warmup

**Nei 3 notebook di training (celle non toccate nel Blocco 1):**
8. ☐ Passare `amp_dtype=AMP_DTYPE` (e `scaler=None` per bf16) alle chiamate `train_one_epoch(...)` nelle celle Phase 1 e Phase 2
9. ☐ Aggiungere `persistent_workers=True, prefetch_factor=4` alle chiamate `DataLoader(...)`

**Negli script di valutazione (devono combaciare col training):**
10. ☐ [src/evaluate_3d_test.py:72](src/evaluate_3d_test.py#L72) — `ENCODER` allineato all'encoder di training
11. ☐ [src/evaluate_3d_test.py:131](src/evaluate_3d_test.py#L131) — variante SegFormer allineata

**Flag globali consigliate (cella di setup notebook, su A100):**
12. ☐ `torch.backends.cuda.matmul.allow_tf32 = True` e `torch.backends.cudnn.allow_tf32 = True` — accelerano le matmul su Ampere. ⚠️ Coesistono col determinismo cuDNN del fix §6.3 (TF32 non rompe `cudnn.deterministic`), ma introducono una minima differenza numerica rispetto all'fp32 puro: accettabile per il training, da documentare.

---

## 10. Principi di scaling rispettati

1. **Nessuno stravolgimento architetturale:** tutte le modifiche sono cambi di *valore* (config) o varianti *drop-in* (encoder/backbone). La pipeline 2D slice-based, la loss, la schedule two-phase e il formato dei tensori restano identici.
2. **Un solo punto di controllo:** grazie al fix §2.1, 7 delle ~12 modifiche si fanno in `config.py`. Le restanti sono nei notebook (loop/loader) e negli script di eval, esplicitamente elencate.
3. **Sinergia con i fix del Blocco 1:** crop=240 chiude l'asimmetria §7.1; bf16 sfrutta l'infrastruttura AMP §6.1/§6.2; il seeding §6.3 resta valido con più worker.
4. **Misurare prima di massimizzare:** partire dalla *config A* (smoke test, 1 epoca) per leggere la VRAM reale dal sanity-check, poi salire a B/C. Evita OOM a metà training su Colab.
5. **Coerenza training↔eval:** ogni cambio di encoder/modello va replicato negli script di valutazione, altrimenti i checkpoint non si caricano.
