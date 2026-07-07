"""
scripts/compute_class_freq.py
================================
Calcola le frequenze voxel REALI delle 5 classi native BraTS-PEDs sul SOLO
TRAIN split (data/split_3d.json) e i pesi per-classe da usare nella loss.
Risolve il placeholder VOXEL_FREQ=None di src/constants.py.

Riusa src/class_weights.py (count_voxels_from_dir + weights_from_counts):
STESSA funzione usata anche da notebooks/03_train_3d.ipynb per calcolare i
pesi direttamente su Colab a partire da data/processed_3d/train/ — nessuna
formula duplicata in due punti scollegati.

IMPORTANTE (train-only, niente leakage): i conteggi vengono ristretti
ESPLICITAMENTE ai subject_id elencati in split_3d.json["train"] (passati come
`subjects=` a count_voxels_from_dir) — MAI su TRAIN_DIR per intero, che
contiene tutti e 257 i soggetti raw (train+val+test). Includere val/test nel
calcolo dei pesi sarebbe una fuga di informazione nella loss di training.

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

Uso locale vs Colab
--------------------
Questo script e' pensato per la fase LOCALE (richiede data/raw/ + split_3d.json
gia' presenti sul disco locale). Su Colab NON serve eseguirlo ne' caricare
region_class_weights.json a mano su Drive: notebooks/03_train_3d.ipynb calcola
i pesi direttamente su data/processed_3d/train/ (stesse funzioni di questo
modulo) subito dopo la decompressione dei dati, garantendo train-only per
costruzione (quella cartella contiene solo soggetti train).

Uso:
    .venv/Scripts/python scripts/compute_class_freq.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.class_weights import count_voxels_from_dir, weights_from_counts  # noqa: E402

TRAIN_DIR = os.path.join("data", "raw", "BraTS-PEDs-v1", "Training")
SPLIT_PATH = os.path.join("data", "split_3d.json")
FREQ_OUT_PATH = os.path.join("data", "class_freq_3d.json")
WEIGHTS_OUT_PATH = os.path.join("data", "region_class_weights.json")

# Label native a 5 classi (src/constants.py): 0=BG, 1=ET, 2=NET, 3=CC, 4=ED
NUM_CLASSES = 5
CLASS_NAMES = ("background", "ET", "NET", "CC", "ED")


def main() -> None:
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_subjects = split["train"]
    print(f"Soggetti train: {len(train_subjects)} (SOLO train — val/test esclusi)")

    counts = count_voxels_from_dir(TRAIN_DIR, num_classes=NUM_CLASSES, subjects=train_subjects)
    freq = counts / counts.sum()
    weights = weights_from_counts(counts)

    print("\n--- Frequenze voxel per classe (TRAIN ONLY, 5 classi native) ---")
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
          f"(copialo in --data-root prima del training, SE non usi il calcolo inline "
          f"gia' incluso in notebooks/03_train_3d.ipynb per Colab).")


if __name__ == "__main__":
    main()
