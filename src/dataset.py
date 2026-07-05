"""
src/dataset.py
===============
I/O e validazione a livello di soggetto per BraTS-PEDs (pipeline 3D).

Responsabilita' di questo modulo — SOLO I/O, NESSUN preprocessing numerico:
    - individuare i file NIfTI di un soggetto (4 modalita' + segmentazione)
    - caricarli come array NumPy grezzi (nessun clip, nessuna normalizzazione)
    - validare che le etichette di segmentazione siano nella convenzione
      pediatrica a 5 classi attesa (si veda src/constants.py)

La normalizzazione delle intensita' (clip percentile + z-score) NON vive qui:
nella pipeline 3D e' applicata on-the-fly da monai.transforms durante il
training su Colab (es. NormalizeIntensityd), cosi' la fase locale si limita a
riorganizzare/validare i NIfTI grezzi (roadmap: "Dicotomia Locale vs Cloud").

Segmentazione — 5 classi native, NESSUNA remap:
    0 = BG, 1 = ET, 2 = NET, 3 = CC, 4 = ED  (vedi src/constants.py)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import nibabel as nib
import numpy as np

from .constants import NUM_CLASSES

MODALITIES: List[str] = ["t1c", "t1n", "t2f", "t2w"]


def get_subject_paths(subject_dir: str, subject_id: str) -> dict:
    """Ritorna i path attesi per le 4 modalita' e la segmentazione di un soggetto.

    Args:
        subject_dir: Cartella del soggetto (es. .../Training/BraTS-PED-00001-000).
        subject_id:  Identificatore del soggetto (es. "BraTS-PED-00001-000").

    Returns:
        Dict con chiavi "t1c", "t1n", "t2f", "t2w", "seg" -> path assoluto/relativo.
    """
    paths = {mod: os.path.join(subject_dir, f"{subject_id}-{mod}.nii.gz") for mod in MODALITIES}
    paths["seg"] = os.path.join(subject_dir, f"{subject_id}-seg.nii.gz")
    return paths


def validate_subject_files(subject_dir: str, subject_id: str) -> Tuple[bool, List[str]]:
    """Verifica che tutti i file NIfTI attesi esistano per un soggetto.

    Args:
        subject_dir: Cartella del soggetto.
        subject_id:  Identificatore del soggetto.

    Returns:
        (is_valid, missing_files): is_valid=True se tutti i 5 file esistono;
        missing_files elenca i path assenti (lista vuota se is_valid=True).
    """
    paths = get_subject_paths(subject_dir, subject_id)
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    return len(missing) == 0, missing


def load_subject_raw(
    subject_dir: str,
    subject_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carica le 4 modalita' + segmentazione di un soggetto, SENZA preprocessing.

    Nessun clip, nessuna normalizzazione: restituisce esattamente i valori
    letti dai file NIfTI. La normalizzazione avviene altrove (transform MONAI,
    fase Colab).

    Args:
        subject_dir: Cartella del soggetto.
        subject_id:  Identificatore del soggetto (es. "BraTS-PED-00001-000").

    Returns:
        images: np.ndarray [4, H, W, D] float32 — modalita' nell'ordine MODALITIES,
                valori di intensita' grezzi (nessuna normalizzazione).
        seg:    np.ndarray [H, W, D] int16 — etichette grezze {0,1,2,3,4}.

    Raises:
        FileNotFoundError: se uno dei file NIfTI attesi manca.
    """
    is_valid, missing = validate_subject_files(subject_dir, subject_id)
    if not is_valid:
        raise FileNotFoundError(
            f"File mancanti per il soggetto {subject_id!r}: {missing}"
        )

    paths = get_subject_paths(subject_dir, subject_id)

    imgs: List[np.ndarray] = []
    for mod in MODALITIES:
        vol = nib.load(paths[mod]).get_fdata().astype(np.float32)
        imgs.append(vol)
    images = np.stack(imgs, axis=0)  # [4, H, W, D]

    seg = nib.load(paths["seg"]).get_fdata().astype(np.int16)  # [H, W, D]

    return images, seg


def validate_segmentation_labels(
    seg: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Tuple[bool, List[int]]:
    """Verifica che le etichette di un volume di segmentazione siano valide.

    Args:
        seg:         Array di etichette intere (qualunque shape).
        num_classes: Numero di classi attese (default 5: {0,1,2,3,4}).

    Returns:
        (is_valid, unexpected_labels): is_valid=True se tutte le etichette
        presenti sono in range(num_classes); unexpected_labels elenca i
        valori fuori range (lista vuota se is_valid=True).
    """
    unique_labels = np.unique(seg).astype(int).tolist()
    unexpected = [lbl for lbl in unique_labels if lbl < 0 or lbl >= num_classes]
    return len(unexpected) == 0, unexpected


def load_subject_nifti_meta(subject_dir: str, subject_id: str) -> nib.Nifti1Image:
    """Carica l'immagine di segmentazione NIfTI per accedere ad affine/header.

    Utile per ri-scrivere le predizioni come NIfTI validi (stesso affine del
    soggetto originale) in fase di export/valutazione.

    Args:
        subject_dir: Cartella del soggetto.
        subject_id:  Identificatore del soggetto.

    Returns:
        nibabel.Nifti1Image della segmentazione (accedi a .affine e .header).
    """
    paths = get_subject_paths(subject_dir, subject_id)
    return nib.load(paths["seg"])
