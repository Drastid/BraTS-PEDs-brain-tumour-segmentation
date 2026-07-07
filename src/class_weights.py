"""
src/class_weights.py
=======================
Utility CONDIVISE per calcolare le frequenze voxel e i pesi per-classe (Eq.
13 del paper GSL, Celaya et al.) a partire da una cartella di soggetti gia'
organizzata (una sottocartella per subject_id, ciascuna con
<subject_id>-seg.nii.gz — convenzione src/dataset_3d.py::build_subject_dicts).

Usato da DUE punti, deliberatamente con la stessa funzione (single source of
truth, per non ricalcolare la formula due volte in modo scollegato):
    - scripts/compute_class_freq.py — fase LOCALE, su data/raw/ + split_3d.json
      (restringe esplicitamente ai soli subject_id del train split).
    - notebooks/03_train_3d.ipynb — fase Colab, chiamata DIRETTAMENTE su
      data/processed_3d/train/ subito dopo la decompressione. In questo caso
      NON serve passare la lista dei soggetti: quella cartella contiene GIA'
      solo i soggetti train (ce li ha messi scripts/reorganize_volumes.py
      seguendo split_3d.json), quindi non c'e' alcun rischio di leakage
      val/test nei pesi, e non serve alcun file precalcolato caricato a mano
      su Drive.

Nessuna dipendenza da torch/monai: puro numpy + nibabel, eseguibile sia in
locale sia su Colab prima ancora di aver importato il resto della pipeline.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import nibabel as nib
import numpy as np


def count_voxels_from_dir(
    subjects_dir: str,
    num_classes: int = 5,
    subjects: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Somma i voxel per classe su (un sottoinsieme del)le sottocartelle soggetto
    di `subjects_dir`.

    Args:
        subjects_dir: Cartella con una sottocartella per soggetto, ciascuna
                      contenente <subject_id>-seg.nii.gz.
        num_classes:  Numero di classi native (5: BG, ET, NET, CC, ED).
        subjects:     Se fornito, limita il conteggio a questi subject_id
                      (es. la lista "train" di split_3d.json — caso fase
                      LOCALE, dove subjects_dir contiene TUTTI i soggetti
                      raw). Se None, usa TUTTE le sottocartelle presenti in
                      subjects_dir (caso Colab: la cartella e' gia' filtrata
                      per split, es. data/processed_3d/train/ — ogni
                      sottocartella li' e' per definizione un soggetto train).

    Returns:
        np.ndarray shape (num_classes,) coi conteggi voxel totali, sommati
        su tutti i soggetti considerati.
    """
    if subjects is None:
        subjects = sorted(
            d for d in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, d))
        )
    counts = np.zeros(num_classes, dtype=np.int64)
    for subj in subjects:
        seg_path = os.path.join(subjects_dir, subj, f"{subj}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        counts += np.bincount(seg.ravel(), minlength=num_classes)[:num_classes]
    return counts


def weights_from_counts(counts: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """w_k ∝ 1/n_k, normalizzati a somma 1 (Eq. 13 del paper GSL — stessa
    formula di src/losses_3d.py::compute_global_class_weights, reimplementata
    qui in numpy puro per non dipendere da torch nella fase LOCALE/Colab
    pre-training).
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    inv = 1.0 / (counts_arr + eps)
    return inv / inv.sum()
