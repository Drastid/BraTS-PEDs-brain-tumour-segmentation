"""
src/constants.py
================
Costanti condivise per la pipeline BraTS-PEDs 3D.

A differenza del vecchio progetto 2D (master/src/constants.py), qui le classi
sono 5, non 4: BraTS-PEDs non segue la convenzione BraTS-adulti (0=BG, 1=NCR,
2=ED, 4=ET vecchio / 3=ET nuovo). La convenzione pediatrica ufficiale e':

    0 = BG  (background)
    1 = ET  (Enhancing Tumor)
    2 = NET (Non-Enhancing Tumor — sostituisce il NCR adulto)
    3 = CC  (Cystic Component — esclusivo pediatrico)
    4 = ED  (Peritumoral Edema)

Verifica empirica (scripts/verify_label_mapping.py, 257 soggetti training):
27 soggetti (10.5%) hanno le label 3 e 4 co-presenti come regioni distinte
nello stesso volume. Fondere 4->3 (come faceva il vecchio progetto, copiando
un pattern BraTS-adulti non applicabile qui) unisce CC ed ED — due regioni
clinicamente separate — in un'unica classe. Qui NON si applica alcuna remap:
si lavora nativamente a 5 classi.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Classi
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 5
CLASS_NAMES: Tuple[str, ...] = ("background", "ET", "NET", "CC", "ED")

# ---------------------------------------------------------------------------
# Dimensioni spaziali (default BraTS-PEDs)
# ---------------------------------------------------------------------------

ORIG_SIZE: int = 240   # H e W di ogni volume NIfTI
N_SLICES: int = 155    # profondita' assiale di ogni volume BraTS-PEDs

# ---------------------------------------------------------------------------
# Frequenza voxel per classe (placeholder — vedi scripts/compute_class_freq.py)
# ---------------------------------------------------------------------------
# A differenza del vecchio VOXEL_FREQ (4 valori, statistiche sporche dalla
# remap 4->3), questo vettore va popolato con le frequenze REALI a 5 classi,
# calcolate sul TRAIN split. scripts/compute_class_freq.py fa esattamente
# questo: legge data/split_3d.json, conta i voxel per classe sul train e
# scrive data/class_freq_3d.json (frequenze + pesi) e
# data/region_class_weights.json (pesi pronti per il ramo --loss=dice_focal
# di run_pipeline_3d.py). Resta comunque a None qui: e' un dato
# dipendente dallo split, non una costante statica del modulo — i chiamanti
# devono leggerlo a runtime dal JSON generato, mai assumere questo fallback
# silenziosamente.
VOXEL_FREQ: np.ndarray | None = None
