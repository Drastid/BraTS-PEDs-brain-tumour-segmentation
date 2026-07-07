"""
scripts/compute_class_freq.py
================================
Calcola le frequenze voxel REALI delle 5 classi native BraTS-PEDs sul TRAIN
split (data/split_3d.json) e i pesi per-classe da usare nella loss. Risolve
il placeholder VOXEL_FREQ=None di src/constants.py (finora mai calcolato:
si veda il commento li' per il contesto).

Motivazione
-----------
Diagnosi: il modello degrada perche' la classe CC (Cystic Component, indice
3) e' la sub-regione pediatrica piu' rara — questo script la quantifica
(invece di ipotizzarla) contando i voxel reali sul train split, e produce
pesi pronti da passare come class_weights alla loss.

Perche' un file SEPARATO da gsl_class_weights.json
-----------------------------------------------------
Il ramo --loss=gsl di run_pipeline_3d.py legge gia' <data-root>/
gsl_class_weights.json se presente. Per NON alterare il comportamento della
GSL — l'utente la vuole invariata come baseline di confronto — questo
script scrive un file diverso, data/region_class_weights.json, usato SOLO
dal ramo --loss=dice_focal (src/losses_monai.py::build_dice_focal_loss via
run_pipeline_3d.py). Stesso schema {"weights": [...]} di gsl_class_weights.json:
se in futuro si vuole pesare anche la GSL basta copiare/rinominare il file.

Metodologia
-----------
1. Legge la lista dei soggetti TRAIN da data/split_3d.json (nessuna fuga di
   informazione da val/test nel calcolo dei pesi).
2. Per ciascun soggetto conta i voxel delle 5 classi native (0=BG, 1=ET,
   2=NET, 3=CC, 4=ED) dalla segmentazione grezza in
   data/raw/BraTS-PEDs-v1/Training/<subject_id>/<subject_id>-seg.nii.gz.
3. Somma i conteggi su TUTTI i soggetti train (non media per-soggetto, cosi'
   i soggetti con piu' voxel tumorali pesano di piu' nella statistica
   aggregata — coerente con come la loss vede i voxel durante il training).
4. Converte i conteggi in pesi con la STESSA Eq. 13 (w_k ∝ 1/n_k, normalizzati
   a somma 1) gia' validata e usata per la GSL
   (src/losses_3d.py::compute_global_class_weights) — i due rami di loss
   condividono quindi la stessa convenzione di pesatura, anche se applicata
   a file/branch diversi.

Output
------
data/class_freq_3d.json       — diagnostica completa (conteggi, frequenze %,
                                 pesi), per ispezione/logging.
data/region_class_weights.json — {"weights": [w_bg, w_et, w_net, w_cc, w_ed]},
                                 pronto per essere copiato in data_root e
                                 caricato dal ramo --loss=dice_focal.

NOTA: se --data-root del training (Colab) e' una cartella diversa da
data/ locale (tipicamente data/processed_3d/, riempita da
scripts/reorganize_volumes.py e poi caricata su Colab), ricordati di
copiare anche region_class_weights.json in quella cartella prima
dell'upload — esattamente come gia' richiesto per gsl_class_weights.json.

Uso:
    .venv/Scripts/python scripts/compute_class_freq.py
"""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

TRAIN_DIR = os.path.join("data", "raw", "BraTS-PEDs-v1", "Training")
SPLIT_PATH = os.path.join("data", "split_3d.json")
FREQ_OUT_PATH = os.path.join("data", "class_freq_3d.json")
WEIGHTS_OUT_PATH = os.path.join("data", "region_class_weights.json")

# Label native a 5 classi (src/constants.py): 0=BG, 1=ET, 2=NET, 3=CC, 4=ED
NUM_CLASSES = 5
CLASS_NAMES = ("background", "ET", "NET", "CC", "ED")


def compute_global_class_weights(voxel_counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """w_k ∝ 1/n_k, normalizzati a somma 1 (Eq. 13 del paper GSL — identica a
    src/losses_3d.py::compute_global_class_weights, reimplementata qui in
    numpy puro per non dipendere da torch/monai nella fase LOCALE)."""
    counts = np.asarray(voxel_counts, dtype=np.float64)
    inv = 1.0 / (counts + eps)
    return inv / inv.sum()


def count_train_voxels(train_subjects: list[str]) -> np.ndarray:
    """Somma i conteggi voxel per classe su tutti i soggetti TRAIN."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for subj in tqdm(train_subjects, desc="Counting voxels (train)"):
        seg_path = os.path.join(TRAIN_DIR, subj, f"{subj}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        counts += np.bincount(seg.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts


def main() -> None:
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_subjects = split["train"]
    print(f"Soggetti train: {len(train_subjects)}")

    counts = count_train_voxels(train_subjects)
    freq = counts / counts.sum()
    weights = compute_global_class_weights(counts)

    print("\n--- Frequenze voxel per classe (TRAIN, 5 classi native) ---")
    for name, c, f, w in zip(CLASS_NAMES, counts, freq, weights):
        print(f"  {name:10s}  voxel={int(c):>13d}  freq={100 * f:7.4f}%  weight={w:.4f}")

    freq_dict = {
        "train_subjects": len(train_subjects),
        "class_names": list(CLASS_NAMES),
        "voxel_counts": counts.tolist(),
        "voxel_freq": freq.tolist(),
        "weights": weights.tolist(),
    }
    os.makedirs(os.path.dirname(FREQ_OUT_PATH), exist_ok=True)
    with open(FREQ_OUT_PATH, "w") as f:
        json.dump(freq_dict, f, indent=2)
    with open(WEIGHTS_OUT_PATH, "w") as f:
        json.dump({"weights": weights.tolist()}, f, indent=2)

    print(f"\n{FREQ_OUT_PATH} salvato (diagnostica completa).")
    print(f"{WEIGHTS_OUT_PATH} salvato — usato da run_pipeline_3d.py --loss dice_focal "
          f"(copialo in --data-root prima del training su Colab).")


if __name__ == "__main__":
    main()
