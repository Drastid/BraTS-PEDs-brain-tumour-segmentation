"""
scripts/verify_label_mapping.py
================================
Verifica indipendente (roadmap §1.2): il vecchio progetto 2D (master/) applica

    seg = np.where(seg_raw == 4, 3, seg_raw)

giustificandolo come "unione della vecchia/nuova convenzione BraTS ET". Questo
script dimostra se tale remap e' distruttiva per BraTS-PEDs, contando per
ciascun soggetto quali etichette uniche compaiono nella maschera di
segmentazione grezza.

Se un numero significativo di soggetti presenta ENTRAMBE le etichette 3 e 4
nello stesso volume, la remap fonde due regioni tumorali distinte in una sola
classe, distruggendo informazione clinica (CC vs ED secondo la convenzione
BraTS-PEDs: 0=BG, 1=ET, 2=NET, 3=CC, 4=ED).

Uso:
    .venv/Scripts/python scripts/verify_label_mapping.py
"""

from __future__ import annotations

import os
from collections import Counter

import nibabel as nib
import numpy as np

TRAIN_DIR = os.path.join("data", "raw", "BraTS-PEDs-v1", "Training")


def main() -> None:
    subjects = sorted(os.listdir(TRAIN_DIR))
    print(f"Soggetti trovati: {len(subjects)}")

    combo_counter: Counter[tuple[int, ...]] = Counter()
    has_3_and_4 = 0
    has_3_only = 0
    has_4_only = 0
    has_neither = 0

    for subj in subjects:
        seg_path = os.path.join(TRAIN_DIR, subj, f"{subj}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        labels = tuple(sorted(int(v) for v in np.unique(seg)))
        combo_counter[labels] += 1

        has_3 = 3 in labels
        has_4 = 4 in labels
        if has_3 and has_4:
            has_3_and_4 += 1
        elif has_3:
            has_3_only += 1
        elif has_4:
            has_4_only += 1
        else:
            has_neither += 1

    print("\n--- Combinazioni di etichette uniche per soggetto ---")
    for combo, count in sorted(combo_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {combo}: {count} soggetti")

    print("\n--- Riepilogo label 3 vs label 4 ---")
    print(f"  Soggetti con SIA label 3 CHE label 4 (regioni distinte): {has_3_and_4}")
    print(f"  Soggetti con SOLO label 3                              : {has_3_only}")
    print(f"  Soggetti con SOLO label 4                              : {has_4_only}")
    print(f"  Soggetti senza label 3 ne' 4                           : {has_neither}")

    total = len(subjects)
    print(f"\n  Totale soggetti: {total}")
    if has_3_and_4 > 0:
        print(
            f"\n  ==> CONFERMATO: {has_3_and_4}/{total} soggetti "
            f"({100 * has_3_and_4 / total:.1f}%) hanno la label 3 (CC) e la "
            f"label 4 (ED) come regioni DISTINTE nello stesso volume.\n"
            f"      La remap 4->3 del vecchio progetto fonde queste due "
            f"regioni cliniche separate ed e' quindi distruttiva."
        )
    else:
        print(
            "\n  ==> Nessun soggetto ha entrambe le label 3 e 4 "
            "contemporaneamente: servono ulteriori verifiche prima di "
            "concludere che la remap sia sbagliata."
        )


if __name__ == "__main__":
    main()
