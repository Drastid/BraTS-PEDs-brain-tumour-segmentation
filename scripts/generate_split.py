"""
scripts/generate_split.py
==========================
Genera lo split train/val/test per BraTS-PEDs, stratificato per presenza-ET e
quartile di tumour-burden, usando le etichette NATIVE a 5 classi (nessuna
remap 4->3 — si veda src/constants.py e scripts/verify_label_mapping.py).

Perche' NON si riusa master/split.json
---------------------------------------
Il vecchio split (master/split.json) e' stato generato calcolando "ET presence"
sulla label 3 DOPO la remap 4->3, quindi quella label mescolava il vero ET con
parte della CC/ED fusa. Con le 5 classi native, "ET" e' la label 1 pura: la
statistica di stratificazione cambia, quindi lo split va rigenerato da zero
per essere metodologicamente corretto, non semplicemente ricopiato.

Metodologia (fedele alla Strategia C del vecchio progetto, §5-8 di
02_preprocessing.ipynb, ma con conteggi a 5 classi):
    1. Per ciascun soggetto, conta i voxel di ogni classe tumorale nativa
       (ET=1, NET=2, CC=3, ED=4) dalla maschera di segmentazione grezza.
    2. tumour_burden = n_ET + n_NET + n_CC + n_ED (tutti i voxel non-BG).
    3. quartili di tumour_burden (0..3) via np.digitize sui percentili 25/50/75.
    4. et_presence = tumour_burden ET (label 1 pura) > 0.
    5. combined_label = et_presence * 4 + quartile  (fino a 8 bin).
    6. train_test_split stratificato su combined_label: 80% train, 10% val,
       10% test, seed=42 (stesso seed del vecchio progetto, per coerenza
       metodologica anche se i soggetti assegnati possono differire).

Output:
    data/split_3d.json — stessa struttura di master/split.json (train/val/test
    liste di subject_id + metadata), cosi' da poter essere confrontato/ispezionato
    con lo stesso formato. Verra' successivamente usato anche per ri-valutare il
    vecchio progetto 2D con uno split metodologicamente coerente.

Uso:
    .venv/Scripts/python scripts/generate_split.py
"""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TRAIN_DIR = os.path.join("data", "raw", "BraTS-PEDs-v1", "Training")
OUT_PATH = os.path.join("data", "split_3d.json")
SEED = 42

# Label native a 5 classi (src/constants.py): 0=BG, 1=ET, 2=NET, 3=CC, 4=ED
LABEL_ET = 1


def compute_tumour_stats(subjects: list[str]) -> list[dict]:
    """Conta i voxel per classe tumorale nativa per ciascun soggetto."""
    stats = []
    for subj in tqdm(subjects, desc="Computing tumour stats"):
        seg_path = os.path.join(TRAIN_DIR, subj, f"{subj}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int16)

        n_et = int((seg == 1).sum())
        n_net = int((seg == 2).sum())
        n_cc = int((seg == 3).sum())
        n_ed = int((seg == 4).sum())
        n_tumour = n_et + n_net + n_cc + n_ed

        stats.append(
            {
                "subject_id": subj,
                "n_et": n_et,
                "n_net": n_net,
                "n_cc": n_cc,
                "n_ed": n_ed,
                "n_tumour": n_tumour,
                "has_et": n_et > 0,
            }
        )
    return stats


def main() -> None:
    subjects = sorted(os.listdir(TRAIN_DIR))
    print(f"Soggetti trovati: {len(subjects)}")

    stats = compute_tumour_stats(subjects)
    tumour_vols = np.array([s["n_tumour"] for s in stats])
    et_presence = np.array([s["has_et"] for s in stats])

    print(f"\nTumour volume (5-classi nativo) — min: {tumour_vols.min()}  "
          f"max: {tumour_vols.max()}  mean: {tumour_vols.mean():.0f}")
    print(f"Soggetti con ET (label 1 pura) presente: {et_presence.sum()} / "
          f"{len(et_presence)} ({100 * et_presence.mean():.1f}%)")

    # --- Stratificazione: presenza-ET + quartile tumour-burden (Strategia C) ---
    indices = list(range(len(stats)))
    quartiles = np.digitize(tumour_vols, np.percentile(tumour_vols, [25, 50, 75]))
    combined_label = (et_presence.astype(int) * 4 + quartiles).tolist()

    train_idx, tmp_idx, _, tmp_labels = train_test_split(
        indices, combined_label, test_size=0.20, stratify=combined_label, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=tmp_labels, random_state=SEED
    )

    chosen_train = [stats[i]["subject_id"] for i in train_idx]
    chosen_val = [stats[i]["subject_id"] for i in val_idx]
    chosen_test = [stats[i]["subject_id"] for i in test_idx]

    # --- Verifica qualita' dello split ---
    def _split_report(name: str, idx: list[int]) -> None:
        vols = tumour_vols[idx]
        et = et_presence[idx]
        print(f"  {name:5s} ({len(idx):3d}): tumour {vols.mean():7.0f} ± {vols.std():6.0f}  "
              f"ET rate {100 * et.mean():5.1f}%")

    print("\n--- Qualita' dello split (5 classi native) ---")
    _split_report("Train", train_idx)
    _split_report("Val", val_idx)
    _split_report("Test", test_idx)

    split_dict = {
        "strategy": "C — Stratified by ET presence (native label 1) + tumour burden quartile, 5-class scheme",
        "label_scheme": {"0": "background", "1": "ET", "2": "NET", "3": "CC", "4": "ED"},
        "seed": SEED,
        "train": chosen_train,
        "val": chosen_val,
        "test": chosen_test,
        "sizes": {"train": len(chosen_train), "val": len(chosen_val), "test": len(chosen_test)},
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(split_dict, f, indent=2)

    print(f"\n{OUT_PATH} salvato.")
    print(f"  Train: {len(chosen_train)} soggetti")
    print(f"  Val:   {len(chosen_val)} soggetti")
    print(f"  Test:  {len(chosen_test)} soggetti")


if __name__ == "__main__":
    main()
