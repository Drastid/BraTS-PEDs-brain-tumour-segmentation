"""
src/dataset_3d.py
====================
Dataset e transform pipeline MONAI per il training 3D on-the-fly (roadmap
§5.2, fase Colab). Riceve i volumi NIfTI grezzi riorganizzati localmente in
data/processed_3d/{train,val,test}/<subject_id>/ (si veda src/dataset.py per
la fase di I/O/validazione locale) e applica, interamente su Colab:

    1. Caricamento NIfTI (LoadImaged)
    2. Normalizzazione clip-P99.5 + z-score sui soli voxel non-zero
       (ClipAndNormalizeNonZerod, src/transforms_3d.py — replica esatta del
       vecchio progetto 2D)
    3. SOLO in training: patch sampling bilanciato tumore/background
       (RandCropByPosNegLabeld) + augmentation geometrica random (flip,
       rotazione) + augmentation di intensita' z-score-safe
    4. SOLO se richiesta la GSL: calcolo della DTM sulla patch gia' croppata
       (ComputeDistanceMapd, src/transforms.py)

In validation/test NON si applica patch sampling: il volume intero viene
passato a sliding_window_inference (src/train_3d.py), coerente con la
Dicotomia Locale/Cloud e con road_3D.md §5.3.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from .transforms import ComputeDistanceMapd
from .transforms_3d import ClipAndNormalizeNonZerod

MODALITIES: List[str] = ["t1c", "t1n", "t2f", "t2w"]


def build_subject_dicts(split_dir: str) -> List[dict]:
    """Costruisce la lista di dict {"image": [4 path], "label": path} per MONAI.

    Args:
        split_dir: Cartella dello split (es. data/processed_3d/train), con
                   una sottocartella per soggetto contenente i 5 NIfTI grezzi
                   (4 modalita' + seg), prodotta da
                   scripts/reorganize_volumes.py.

    Returns:
        Lista di dizionari, uno per soggetto, nel formato atteso da
        monai.transforms.LoadImaged con keys=["image","label"]: "image" è
        una lista di 4 path (una per modalita', nell'ordine di MODALITIES,
        che LoadImaged impila automaticamente lungo un nuovo asse canale),
        "label" è il path della segmentazione.
    """
    subjects = sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    data = []
    for subj in subjects:
        subj_dir = os.path.join(split_dir, subj)
        image_paths = [os.path.join(subj_dir, f"{subj}-{mod}.nii.gz") for mod in MODALITIES]
        label_path = os.path.join(subj_dir, f"{subj}-seg.nii.gz")
        data.append({"image": image_paths, "label": label_path, "subject_id": subj})
    return data


def build_train_transforms(
    roi: Sequence[int],
    num_classes: int,
    num_samples: int = 2,
    with_dtm: bool = False,
) -> Compose:
    """Pipeline di transform per il TRAINING (patch sampling + augmentation).

    Args:
        roi:         Dimensione della patch 3D (es. (128,128,128)).
        num_classes: Numero di classi (5, per ComputeDistanceMapd).
        num_samples: Numero di patch campionate per volume ad ogni chiamata
                     (RandCropByPosNegLabeld) — piu' di 1 ammortizza il costo
                     di I/O/decodifica NIfTI su piu' patch per volume caricato.
        with_dtm:    Se True, aggiunge ComputeDistanceMapd DOPO il patch
                     sampling (richiesto solo dal ramo --loss=gsl).

    Returns:
        monai.transforms.Compose pronta per Dataset/CacheDataset.
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ClipAndNormalizeNonZerod(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        # Augmentation di intensita' z-score-safe (moltiplicativa/additiva su
        # dati gia' normalizzati mu~0 sigma~1) — coerente con la policy del
        # vecchio progetto ("solo augmentation che non rompe la normalizzazione").
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]
    if with_dtm:
        transforms.append(
            ComputeDistanceMapd(keys=["label"], num_classes=num_classes)
        )
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def build_eval_transforms(num_classes: int, with_dtm: bool = False) -> Compose:
    """Pipeline di transform per VALIDATION/TEST — NESSUN patch sampling.

    Il volume intero (normalizzato) viene passato cosi' com'e' a
    sliding_window_inference nel training loop (src/train_3d.py). with_dtm e'
    incluso per completezza di interfaccia ma nella pratica evaluate_gsl_3d
    non ne ha bisogno (la GSL e' un termine di TRAINING; in validazione si
    riportano solo Dice/HD95, si veda src/metrics_3d.py).

    Args:
        num_classes: Numero di classi (per l'eventuale ComputeDistanceMapd).
        with_dtm:    Se True, calcola la DTM anche in validazione (sul volume
                     intero — piu' costoso; usarlo solo se strettamente
                     necessario per loggare la componente GSL anche in val).

    Returns:
        monai.transforms.Compose.
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ClipAndNormalizeNonZerod(keys=["image"]),
    ]
    if with_dtm:
        transforms.append(ComputeDistanceMapd(keys=["label"], num_classes=num_classes))
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def build_dataloaders_3d(
    data_root: str,
    roi: Sequence[int],
    num_classes: int,
    batch_size: int = 2,
    num_samples: int = 2,
    num_workers: int = 4,
    with_dtm: bool = False,
    use_cache: bool = True,
    cache_rate: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """Costruisce i DataLoader MONAI per train e validation.

    Args:
        data_root:   Root con le sottocartelle "train"/"val" (es.
                     data/processed_3d su Colab, dopo estrazione dell'archivio).
        roi:         Dimensione della patch di training (vincoli architetturali:
                     si veda src/models3d.py — 128 soddisfa sia SwinUNETR
                     (divisibile per 32) sia SegResNet (per 16)).
        num_classes: Numero di classi (5).
        batch_size:  Batch size di training (tipicamente 1-4 su A100 per patch
                     128^3, si veda road_3D.md §5.4).
        num_samples: Patch per volume campionate ad ogni draw di training.
        num_workers: Worker CPU del DataLoader (calcolano transform/DTM in
                     parallelo al forward/backward GPU del batch precedente).
        with_dtm:    Se True, aggiunge ComputeDistanceMapd (richiesto da
                     --loss=gsl).
        use_cache:   Se True, usa CacheDataset per il train set (tiene i
                     volumi DECODIFICATI — pre patch-sampling — in RAM tra le
                     epoche, evitando di ri-decodificare i NIfTI ogni volta).
        cache_rate:  Frazione del train set da cachare (1.0 = tutto, riducibile
                     se la RAM di Colab non basta per l'intero train set).

    Returns:
        (train_loader, val_loader).
    """
    train_dicts = build_subject_dicts(os.path.join(data_root, "train"))
    val_dicts = build_subject_dicts(os.path.join(data_root, "val"))

    train_tf = build_train_transforms(roi, num_classes, num_samples=num_samples, with_dtm=with_dtm)
    val_tf = build_eval_transforms(num_classes, with_dtm=False)

    if use_cache:
        train_ds = CacheDataset(data=train_dicts, transform=train_tf, cache_rate=cache_rate)
    else:
        train_ds = Dataset(data=train_dicts, transform=train_tf)
    val_ds = Dataset(data=val_dicts, transform=val_tf)  # volumi interi: cache sconsigliata (RAM)

    # persistent_workers=True evita di ricreare i worker (e ri-scaldare la
    # cache/pipeline) ad ogni epoca — su decine/centinaia di epoche risparmia
    # tempo di startup ripetuto. Valido SOLO con num_workers>0: con 0 worker
    # PyTorch solleva errore, quindi lo si attiva condizionalmente.
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )
    # batch_size=1 in validazione: sliding_window_inference lavora su un volume
    # alla volta (dimensioni diverse tra soggetti non sono garantite uguali).
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        persistent_workers=persistent,
    )

    return train_loader, val_loader
