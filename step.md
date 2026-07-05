# step.md — Transizione Locale → Google Colab (A100)

To-Do list sequenziale per passare dal preprocessing locale (completato) al
training 3D su Colab. Segui le fasi in ordine: non saltare passaggi, anche se
sembrano ovvi.

> **Stato di partenza (verificato):** il preprocessing locale è concluso
> (`data/split_3d.json` + `data/processed_3d/{train,val,test}/`, 257 soggetti,
> 25GB). **Il codice sorgente per il training 3D è già scritto e testato in
> locale** (Punti 3–9 della roadmap, 32/32 test pytest passati): `src/dataset_3d.py`,
> `src/models3d.py`, `src/losses_3d.py`, `src/losses_monai.py`,
> `src/metrics_3d.py`, `src/optim_3d.py`, `src/train_3d.py`, `src/eval_3d.py`,
> `run_pipeline_3d.py`. Su Colab **non si scrive codice nuovo**: si esegue
> quello già presente nel repository GitHub. L'unico file ancora da creare è
> il notebook-launcher `03_train_3d.ipynb` (Fase 2).
>
> **Il progetto è versionato su GitHub.** Il `.gitignore` esclude
> `data/`, `weights/`, `checkpoints/`, `evaluation_outputs/`, `*.pth`, `*.pt`
> — quindi `git clone` porta SOLO codice (`src/`, `notebooks/*.ipynb`,
> `run_pipeline_3d.py`, `requirements.txt`, i `.md`), **mai** dataset o pesi.
> Questo cambia la Fase 1: non serve più zippare/caricare `src/` su Drive —
> il codice arriva via `git clone` direttamente su Colab (Fase 2). Drive resta
> necessario SOLO per i dati (grandi, esclusi dal repo) e per i checkpoint di
> output (persistenza fra sessioni Colab effimere).

---

## Fase 1 — Preparazione e Upload (azione manuale tua, in locale)

Il codice **non** passa da qui: arriverà su Colab via `git clone` (Fase 2).
Questa fase serve solo per i dati (esclusi dal repo da `.gitignore`) e per
predisporre la cartella di output su Drive.

- [ ] **1.1** Verifica lo spazio libero: i tre split pesano circa
      `train=20GB`, `val=2.7GB`, `test=2.6GB` (~25GB totali). Assicurati di
      avere almeno 50GB liberi in locale (per gli zip temporanei) e altrettanti
      su Google Drive.

- [ ] **1.2** Crea gli archivi zip, **uno per split**, dalla cartella
      `BraTS-PEDs-3D/data/processed_3d/`:
      ```bash
      cd "BraTS-PEDs-3D/data/processed_3d"
      zip -r train_3d.zip train
      zip -r val_3d.zip val
      zip -r test_3d.zip test
      ```
      Nomi esatti dei file da produrre: `train_3d.zip`, `val_3d.zip`,
      `test_3d.zip`. Non rinominarli diversamente: i comandi della Fase 2
      assumono questi nomi esatti.

- [ ] **1.3** Il file di split (`data/split_3d.json`) resta un file singolo,
      **non va zippato** — è piccolo (pochi KB) e va caricato così com'è.

- [ ] **1.4** Apri Google Drive (drive.google.com) e ricrea **esattamente**
      questa alberatura (creala manualmente se non esiste già):
      ```
      Drive/
      └── BraTS_Project/
          ├── data/
          │   ├── split_3d.json
          │   ├── train_3d.zip
          │   ├── val_3d.zip
          │   └── test_3d.zip
          └── checkpoints/          (cartella vuota, creala ora: ci scriverà il training)
      ```
      Nota: non serve più una cartella `code/` — il codice arriva via
      `git clone` direttamente su Colab (Fase 2), non da Drive. `checkpoints/`
      deve esistere già vuota **prima** di lanciare il training, perché
      `run_pipeline_3d.py --backup-dir` copia lì i pesi alla fine di ogni run
      — se la cartella non esiste, la creazione su Drive può fallire
      silenziosamente per permessi di sync.

- [ ] **1.5** Carica i 4 file (`split_3d.json`, `train_3d.zip`, `val_3d.zip`,
      `test_3d.zip`) nella sottocartella `Drive/BraTS_Project/data/`. Attendi
      che la sincronizzazione di Drive sia completa (icona di sync ferma)
      prima di aprire Colab — un upload incompleto produce zip corrotti che
      falliscono silenziosamente in fase di unzip.

- [ ] **1.6** (Facoltativo ma consigliato) Se hai già scelto la fonte dei
      pesi pre-addestrati (es. checkpoint SwinUNETR da HuggingFace, o pesi SSL
      NVIDIA `model_swinvit.pt`), carica anche quel file `.pt`/`.pth` in
      `Drive/BraTS_Project/pretrained/`. Se non l'hai ancora deciso, salta
      questo punto: lo smoke test in Fase 4 funziona anche senza
      (`--pretrained none`).

- [ ] **1.7** Assicurati che il push su GitHub sia stato effettuato e che il
      repository sia visibile (pubblico, oppure privato con un Personal
      Access Token pronto — serve in Fase 2.3 se il repo è privato). Annota
      l'URL esatto del repo, es. `https://github.com/<tuo-utente>/BraTS-PEDs-3D.git`.

---

## Fase 2 — Il Notebook Colab "Launcher" (`03_train_3d.ipynb`)

Questo notebook va creato su Colab (non in locale) con le seguenti celle, in
questo ordine esatto.

- [ ] **2.1 — Cella 1: verifica GPU**
      ```python
      !nvidia-smi
      ```
      Controlla che appaia una A100 (`Runtime > Change runtime type > A100 GPU`
      se non è già selezionata). Se appare una T4/L4, cambia runtime PRIMA di
      proseguire: il roadmap assume A100 per batch size e ROI configurati.

- [ ] **2.2 — Cella 2: mount di Google Drive**
      ```python
      from google.colab import drive
      drive.mount('/content/drive')
      ```
      Autorizza l'accesso quando richiesto dal popup.

- [ ] **2.3 — Cella 3: clone del repository GitHub (codice)**
      ```python
      import os

      REPO_URL = "https://github.com/<tuo-utente>/BraTS-PEDs-3D.git"
      LOCAL_ROOT = "/content/BraTS-PEDs-3D"

      !git clone {REPO_URL} {LOCAL_ROOT}
      ```
      Se il repository è **privato**, usa invece un Personal Access Token
      (creato su GitHub in `Settings > Developer settings > Personal access
      tokens`, permesso minimo `repo:read`) inserito nell'URL:
      ```python
      GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"  # non committare/condividere mai questo valore
      REPO_URL = f"https://{GITHUB_TOKEN}@github.com/<tuo-utente>/BraTS-PEDs-3D.git"
      !git clone {REPO_URL} {LOCAL_ROOT}
      ```
      Il clone porta **solo codice** (`src/`, `notebooks/`, `run_pipeline_3d.py`,
      `requirements.txt`, i `.md`): `data/`, `weights/`, `checkpoints/`,
      `evaluation_outputs/` sono esclusi da `.gitignore` e non esistono ancora
      su questa VM — li porta la cella successiva da Drive.

- [ ] **2.4 — Cella 4: copia degli zip da Drive al disco locale di Colab e
      decompressione (SOLO dati, non codice)**
      ```python
      DRIVE_ROOT = "/content/drive/MyDrive/BraTS_Project"
      DATA_ROOT = f"{LOCAL_ROOT}/data"
      os.makedirs(DATA_ROOT, exist_ok=True)

      # Copia split.json
      !cp "{DRIVE_ROOT}/data/split_3d.json" "{DATA_ROOT}/split_3d.json"

      # Copia e decomprimi i dati (train/val/test)
      os.makedirs(f"{DATA_ROOT}/processed_3d", exist_ok=True)
      for split in ["train", "val", "test"]:
          !cp "{DRIVE_ROOT}/data/{split}_3d.zip" "{DATA_ROOT}/"
          !unzip -q "{DATA_ROOT}/{split}_3d.zip" -d "{DATA_ROOT}/processed_3d/"
      ```
      **Perché è vitale:** Google Drive montato via FUSE (`/content/drive/...`)
      ha una latenza I/O molto più alta di un disco locale, e MONAI durante il
      training farà migliaia di letture random di file NIfTI per epoca
      (`RandCropByPosNegLabeld` legge il volume intero ad ogni draw). Leggere
      direttamente da Drive trasforma la GPU A100 in un collo di bottiglia I/O
      bound, con la GPU che resta idle in attesa dei dati — nei test locali
      precedenti, il DataLoader con `num_workers>0` richiede accesso rapido e
      ripetuto ai file. Copiare tutto su `/content/` (SSD locale della VM
      Colab) prima di iniziare elimina completamente questo collo di
      bottiglia. Il costo è ~2-5 minuti di copia una tantum a inizio sessione,
      trascurabile rispetto a ore di training. Lo stesso ragionamento NON si
      applica al codice (`src/`), che è piccolo e viene letto una sola volta
      all'import: per quello `git clone` diretto su `/content/` è già ottimale.

- [ ] **2.5 — Cella 5: verifica integrità post-unzip**
      ```python
      import json

      with open(f"{DATA_ROOT}/split_3d.json") as f:
          split = json.load(f)
      print("sizes dichiarate:", split["sizes"])

      for s in ["train", "val", "test"]:
          d = f"{DATA_ROOT}/processed_3d/{s}"
          n_subj = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])
          print(f"{s}: {n_subj} cartelle soggetto su disco")
      ```
      Deve stampare `train: 205, val: 26, test: 26` — se i numeri non
      corrispondono, uno zip non si è decompresso correttamente: torna al
      punto 2.4.

- [ ] **2.6 — Cella 6: installazione dipendenze**
      ```python
      !pip install -q -r {LOCAL_ROOT}/requirements.txt
      ```
      Nota: `torch` è già preinstallato nell'immagine Colab standard con
      build CUDA — `requirements.txt` fissa solo una versione minima
      (`torch>=2.0`), quindi `pip` non lo reinstallerà se quello presente la
      soddisfa già. Se invece vuoi installare solo le dipendenze aggiuntive
      senza rischiare un downgrade di torch:
      ```python
      !pip install -q "monai>=1.3" "nibabel>=5.4" "einops>=0.8" "huggingface_hub>=0.20"
      ```

- [ ] **2.7 — Cella 7: verifica import del progetto**
      ```python
      import sys
      sys.path.insert(0, LOCAL_ROOT)

      from src.dataset_3d import build_dataloaders_3d
      from src.models3d import build_model_3d, load_pretrained_3d
      from src.losses_3d import DiceFocalGSLLoss, AlphaScheduler
      from src.losses_monai import build_dice_focal_loss
      from src.train_3d import train_one_epoch_3d, evaluate_3d

      print("Import OK.")
      ```

---

## Fase 3 — Stato del codice sorgente (nessuno sviluppo nuovo richiesto)

Il codice per il training 3D **è già stato scritto e testato in locale**
(32/32 test pytest passati, si veda `pipeline3D.md` Punti 3–9). Non è
necessario generare nulla di nuovo prima di poter lanciare un training. Elenco
di riferimento dei file coinvolti, nell'ordine in cui `run_pipeline_3d.py` li
importa ed esegue:

- [x] `src/dataset_3d.py` — `build_subject_dicts`, `build_train_transforms`,
      `build_eval_transforms`, `build_dataloaders_3d`: DataLoader MONAI con
      trasformazioni on-the-fly (normalizzazione `ClipAndNormalizeNonZerod`,
      `RandCropByPosNegLabeld`, augmentation geometriche, `ComputeDistanceMapd`
      opzionale per la DTM della GSL).
- [x] `src/models3d.py` — `build_model_3d` (DynUNet/SegResNet/SwinUNETR),
      `load_pretrained_3d` (caricamento pesi agnostico rispetto alla fonte,
      shape-matching automatico).
- [x] `src/losses_3d.py` — `DiceFocalGSLLoss` (Generalized Surface Loss
      schedulata via `AlphaScheduler`) e `src/losses_monai.py` —
      `build_dice_focal_loss` (baseline MONAI nativa).
- [x] `src/metrics_3d.py` — Dice + HD95 per-sottoregione (4 sub-regioni
      pediatriche), loggate separatamente.
- [x] `src/optim_3d.py` — `build_optimizer_3d` (LR differenziato
      backbone/head), `set_backbone_trainable` (freeze/unfreeze per warm-up).
- [x] `src/train_3d.py` — `train_one_epoch_3d`, `train_one_epoch_gsl_3d`,
      `evaluate_3d`, `save_checkpoint`/`load_checkpoint`, `set_seed`.
- [x] `src/eval_3d.py` — `evaluate_test_set_3d`, `summarize_metrics`:
      valutazione 3D nativa con sliding-window inference, post-processing ed
      export NIfTI per soggetto.
- [x] `run_pipeline_3d.py` — orchestratore end-to-end (CLI unica, si veda
      Fase 4).

**Unico artefatto ancora da creare, e va creato su Colab stesso (Fase 2):**

- [ ] `notebooks/03_train_3d.ipynb` — il notebook-launcher che monta Drive,
      clona il repo GitHub, decomprime i dati, installa le dipendenze e
      invoca `!python run_pipeline_3d.py ...` (Fase 4) dalla sua ultima
      cella. Non è un file di logica di training: è solo l'orchestratore
      dell'ambiente Colab attorno al codice già esistente. Una volta scritto e
      testato su Colab, conviene ricommitarlo nel repo (`git add
      notebooks/03_train_3d.ipynb && git commit && git push` dal tuo locale)
      così le prossime sessioni Colab lo trovano già pronto via `git clone`
      invece di doverlo riscrivere da zero.

Se durante lo smoke test (Fase 4) emergono errori legati a differenze fra
ambiente locale (CPU-only) e Colab (CUDA/A100) — es. dtype AMP, memory
layout, versione di MONAI — quelli sì richiederanno una modifica mirata a uno
dei file sopra. Non è previsto sviluppo *ex novo*, solo eventuale debug
puntuale.

---

## Fase 4 — Esecuzione e Smoke Test

- [ ] **4.1 — Smoke test** (1 epoca, batch size ridotto, ROI piccola, nessun
      backup, per verificare che la pipeline giri senza OOM né errori):
      ```bash
      !cd {LOCAL_ROOT} && python run_pipeline_3d.py \
          --arch dynunet \
          --loss dice_focal \
          --data-root {DATA_ROOT}/processed_3d \
          --roi 64 64 64 \
          --batch-size 1 \
          --num-samples 1 \
          --num-workers 2 \
          --epochs 1 \
          --pretrained none \
          --run-name smoke_test \
          --no-evaluate
      ```
      Cosa verificare nell'output:
      - Nessun `CUDA out of memory`.
      - `PyTorch ... device=cuda ... GPU=A100...` stampato correttamente
        all'avvio (conferma che la GPU è effettivamente in uso, non CPU).
      - Una riga `[ep 1/1] train_loss=... val_dice_fg=... val_hd95_fg=...`
        stampata senza eccezioni.
      - File creati in `{LOCAL_ROOT}/checkpoints/smoke_test/dynunet_dice_focal/`
        (`best.pth`, `last.pth`, `history.json`).

- [ ] **4.2** Ripeti lo smoke test con `--loss gsl` per verificare anche il
      ramo della Generalized Surface Loss (calcolo DTM on-the-fly incluso):
      ```bash
      !cd {LOCAL_ROOT} && python run_pipeline_3d.py \
          --arch dynunet --loss gsl \
          --data-root {DATA_ROOT}/processed_3d \
          --roi 64 64 64 --batch-size 1 --num-samples 1 --num-workers 2 \
          --epochs 1 --pretrained none --run-name smoke_test --no-evaluate
      ```

- [ ] **4.3** Se entrambi gli smoke test passano senza errori, procedi con il
      **training vero** sulla A100, con backup su Drive e valutazione
      automatica finale (default ON), ad esempio per la baseline
      DiceFocalLoss su SegResNet con pesi pre-addestrati. I flag pesi sono
      **specifici per architettura** (`--pretrained-swinunetr`,
      `--pretrained-segresnet` — nessun `--weights-path` condiviso, dato che
      un checkpoint SwinUNETR non ha alcuna chiave in comune con SegResNet):
      ```bash
      !cd {LOCAL_ROOT} && python run_pipeline_3d.py \
          --arch segresnet \
          --loss dice_focal \
          --data-root {DATA_ROOT}/processed_3d \
          --roi 128 128 128 \
          --batch-size 2 \
          --num-samples 2 \
          --num-workers 4 \
          --epochs 100 \
          --pretrained auto \
          --pretrained-segresnet /content/drive/MyDrive/BraTS_Project/pretrained/<nome_file_pesi> \
          --run-name run01 \
          --backup-dir /content/drive/MyDrive/BraTS_Project/checkpoints
      ```
      Se non hai ancora caricato pesi pre-addestrati (punto 1.6 saltato),
      ometti `--pretrained-segresnet`/`--pretrained-swinunetr` e usa
      `--pretrained none` per un training da zero. DynUNet non ha un flag
      pesi dedicato: parte sempre da zero (nessun checkpoint compatibile
      individuato per questa architettura).

      Per allenare tutte e tre le architetture in sequenza (dynunet,
      segresnet, swinunetr), sostituisci `--arch segresnet` con `--all`:
      ciascuna finisce nella propria sottocartella
      `<run-name>/<arch>_<loss>/`, passando `--pretrained-segresnet`/
      `--pretrained-swinunetr` per i rispettivi checkpoint.

- [ ] **4.4** Monitora periodicamente l'esecuzione (Colab disconnette sessioni
      idle): controlla che `--backup-dir` stia effettivamente scrivendo
      `best.pth`/`last.pth`/`history.json` su Drive dopo ogni epoca stampata,
      così da non perdere progressi in caso di disconnessione improvvisa.

- [ ] **4.5** A fine training, verifica `evaluation_outputs/run01/segresnet_dice_focal/test_3d_metrics.json`
      (creato automaticamente da `--evaluate`, default ON) per le metriche
      Dice/HD95 finali per-sottoregione sul test set, e le predizioni NIfTI
      esportate in `evaluation_outputs/run01/segresnet_dice_focal/nifti_predictions/`.

- [ ] **4.6** Se durante uno smoke test trovi un bug da correggere: **non
      editare i file direttamente dentro `/content/BraTS-PEDs-3D/` su Colab**
      (si perdono alla disconnessione). Correggi in locale, `git commit` +
      `git push` dal tuo computer, poi su Colab aggiorna con:
      ```python
      !cd {LOCAL_ROOT} && git pull
      ```
      e ripeti lo smoke test. Questo mantiene GitHub come unica fonte di
      verità del codice, coerente con l'uso di `git clone` in Fase 2.3.

---

Al termine di ogni run, ripeti la Fase 4.3 cambiando `--arch`/`--loss`/
`--run-name` per le altre combinazioni pianificate (es. `swinunetr`+`gsl`,
`dynunet`+`gsl`, ecc.), oppure usa `--all` per allenarle tutte in sequenza in
un solo comando, riutilizzando lo stesso ambiente Colab già preparato nelle
Fasi 1–2 senza doverle rifare (basta un `git pull` se nel frattempo hai
aggiornato il codice su GitHub).
