"""
scripts/reorganize_volumes.py
===============================
Riorganizza i volumi NIfTI grezzi in data/raw/ secondo lo split
train/val/test definito da data/split_3d.json (roadmap §2 — fase LOCALE).

Fase locale vs Colab (Dicotomia Locale/Cloud)
----------------------------------------------
Questo script fa SOLO riorganizzazione: copia fisica delle cartelle soggetto
nella struttura data/processed_3d/{train,val,test}/<subject_id>/, cosi'
com'erano in data/raw/ (4 modalita' + seg, NIfTI grezzi, nessuna modifica).

NON fa:
    - estrazione di patch/slice su disco
    - normalizzazione/clip delle intensita'
    - conversione di formato

L'obiettivo e' produrre una struttura pronta per essere compressa (zip/tar) e
trasferita su Colab, dove le monai.transforms (LoadImaged, NormalizeIntensityd,
RandCropByPosNegLabeld, ...) opereranno on-the-fly durante il training.

Uso:
    .venv/Scripts/python scripts/reorganize_volumes.py
"""

from __future__ import annotations

import json
import os
import shutil

from tqdm import tqdm

RAW_TRAIN_DIR = os.path.join("data", "raw", "BraTS-PEDs-v1", "Training")
SPLIT_PATH = os.path.join("data", "split_3d.json")
OUT_BASE = os.path.join("data", "processed_3d")

# I 5 file NIfTI attesi per soggetto (4 modalita' + segmentazione)
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


def copy_subject(subject_id: str, dst_split_dir: str) -> None:
    """Copia la cartella di un soggetto (5 file NIfTI) nella destinazione."""
    src_dir = os.path.join(RAW_TRAIN_DIR, subject_id)
    dst_dir = os.path.join(dst_split_dir, subject_id)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f"{subject_id}-{mod}.nii.gz" for mod in MODALITIES]
    files.append(f"{subject_id}-seg.nii.gz")

    for fname in files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"File atteso mancante: {src}")
        shutil.copy2(src, dst)


def main() -> None:
    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(
            f"{SPLIT_PATH} non trovato. Esegui prima scripts/generate_split.py."
        )

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    print(f"Split caricato: {split['strategy']}")
    print(f"  Sizes: {split['sizes']}")

    total_copied = 0
    for split_name in ("train", "val", "test"):
        subject_ids = split[split_name]
        dst_split_dir = os.path.join(OUT_BASE, split_name)
        os.makedirs(dst_split_dir, exist_ok=True)

        for subj in tqdm(subject_ids, desc=f"[{split_name:5s}] Copying"):
            copy_subject(subj, dst_split_dir)
            total_copied += 1

        print(f"  {split_name}: {len(subject_ids)} soggetti copiati -> {dst_split_dir}")

    print(f"\nTotale soggetti copiati: {total_copied}")
    print(f"Output: {os.path.abspath(OUT_BASE)}")
    print(
        "\nStruttura risultante (nessuna estrazione/normalizzazione applicata):\n"
        "  data/processed_3d/\n"
        "      train/<subject_id>/<subject_id>-{t1c,t1n,t2f,t2w,seg}.nii.gz\n"
        "      val/...\n"
        "      test/...\n"
    )


if __name__ == "__main__":
    main()
