"""
tests/test_dataset_3d.py
===========================
Suite pytest formale per src/dataset_3d.py (roadmap §9: "test_dataset3d.py:
shape delle patch dopo il patch sampling di training").

Fino a questo punto src/dataset_3d.py era stato verificato solo con script
manuali inline (Punto 5 di pipeline3D.md). Questo file formalizza quelle
verifiche in test permanenti ed eseguibili via `pytest`, aggiungendo inoltre
casi non ancora coperti: patch a dimensioni diverse, gestione di split
vuoti/soggetti con file mancanti, e l'integrità delle etichette nelle patch
dopo le augmentation geometriche.

NOTA: il patch sampling e' passato da RandCropByPosNegLabeld (bilanciamento solo fg/bg) a RandCropByLabelClassesd (bilanciamento per-classe, con CC sovra-pesata — vedi src/dataset_3d.py::DEFAULT_CLASS_SAMPLE_RATIOS e scripts/compute_class_freq.py). I test sotto verificano shape/integrita' e restano validi indipendentemente dalla strategia di sampling usata.

Tutti i test che richiedono dati reali (data/processed_3d/) vengono saltati
automaticamente (pytest.skip) se quella cartella non e' presente in questo
ambiente, cosi' la suite resta eseguibile anche in un checkout senza il
dataset scaricato.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from monai.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_3d import (
    MODALITIES,
    build_dataloaders_3d,
    build_eval_transforms,
    build_subject_dicts,
    build_train_transforms,
)

TEST_DATA_ROOT = os.path.join("data", "processed_3d")
NUM_CLASSES = 5


def _skip_if_no_real_data() -> str:
    if not os.path.isdir(TEST_DATA_ROOT):
        pytest.skip(f"{TEST_DATA_ROOT} non presente in questo ambiente — richiede il dataset reale.")
    return TEST_DATA_ROOT


# ---------------------------------------------------------------------------
# build_subject_dicts
# ---------------------------------------------------------------------------


def test_build_subject_dicts_real_train_split() -> None:
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "train"))

    assert len(dicts) == 205, f"attesi 205 soggetti train, trovati {len(dicts)}"
    first = dicts[0]
    assert "image" in first and "label" in first and "subject_id" in first
    assert len(first["image"]) == len(MODALITIES)
    assert all(os.path.isfile(p) for p in first["image"])
    assert os.path.isfile(first["label"])


def test_build_subject_dicts_empty_split(tmp_path) -> None:
    """Split vuoto (nessuna sottocartella soggetto) -> lista vuota, nessun errore."""
    empty_dir = tmp_path / "empty_split"
    empty_dir.mkdir()

    dicts = build_subject_dicts(str(empty_dir))
    assert dicts == []


def test_build_subject_dicts_missing_files_produce_unreadable_paths(tmp_path) -> None:
    """Una cartella soggetto senza i NIfTI attesi produce comunque un dict (i
    path sono costruiti per convenzione), ma i file non esistono — la
    responsabilita' di fallire e' di LoadImaged a runtime, non di
    build_subject_dicts (che e' pura costruzione di path).
    """
    subj_dir = tmp_path / "split" / "BraTS-PED-FAKE-000"
    subj_dir.mkdir(parents=True)
    # Nessun file .nii.gz creato in questa cartella.

    dicts = build_subject_dicts(str(tmp_path / "split"))
    assert len(dicts) == 1
    assert not any(os.path.isfile(p) for p in dicts[0]["image"])


# ---------------------------------------------------------------------------
# Patch sampling: shape dopo RandCropByLabelClassesd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [(64, 64, 64), (128, 128, 128)], ids=["roi64", "roi128"])
def test_train_transform_patch_shape(roi: tuple[int, int, int]) -> None:
    """Verifica che RandCropByLabelClassesd produca patch della dimensione
    richiesta, per due ROI diverse — copre esplicitamente la richiesta della
    roadmap ('shape delle patch dopo il patch sampling di training')."""
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "train"))[:1]

    tf = build_train_transforms(roi=roi, num_classes=NUM_CLASSES, num_samples=1, with_dtm=False)
    ds = Dataset(data=dicts, transform=tf)
    sample = ds[0][0]

    assert tuple(sample["image"].shape) == (len(MODALITIES), *roi)
    assert tuple(sample["label"].shape) == (1, *roi)


def test_train_transform_num_samples_multiplies_batch() -> None:
    """num_samples=N produce N patch per volume: un DataLoader con
    batch_size=B su un solo volume deve produrre un batch di N*1 item per
    ogni draw (verificato con B=num_samples, un solo soggetto in input)."""
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "train"))[:1]

    num_samples = 3
    roi = (64, 64, 64)
    tf = build_train_transforms(roi=roi, num_classes=NUM_CLASSES, num_samples=num_samples)
    ds = Dataset(data=dicts, transform=tf)

    loader = DataLoader(ds, batch_size=num_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == num_samples


# ---------------------------------------------------------------------------
# Integrità delle etichette nelle patch (dopo augmentation geometriche)
# ---------------------------------------------------------------------------


def test_patch_labels_stay_in_valid_range_after_augmentation() -> None:
    """Le augmentation geometriche (flip/rotate) permutano voxel ma non
    devono MAI introdurre valori di classe fuori da {0,...,NUM_CLASSES-1}
    (es. per interpolazione errata) — verifica diretta di integrità delle
    5 classi native lungo tutta la pipeline di training."""
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "train"))[:3]

    tf = build_train_transforms(roi=(64, 64, 64), num_classes=NUM_CLASSES, num_samples=2)
    ds = Dataset(data=dicts, transform=tf)

    for i in range(len(dicts)):
        samples = ds[i]  # con num_samples=2, ds[i] ritorna una lista di 2 dict
        for sample in samples:
            labels = torch.unique(sample["label"])
            assert torch.all((labels >= 0) & (labels < NUM_CLASSES)), (
                f"Label fuori range trovate: {labels.tolist()}"
            )


# ---------------------------------------------------------------------------
# Ramo DTM (with_dtm=True)
# ---------------------------------------------------------------------------


def test_train_transform_with_dtm_shape() -> None:
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "train"))[:1]

    roi = (48, 48, 48)
    tf = build_train_transforms(roi=roi, num_classes=NUM_CLASSES, num_samples=1, with_dtm=True)
    ds = Dataset(data=dicts, transform=tf)
    sample = ds[0][0]

    assert "distance_map" in sample
    assert tuple(sample["distance_map"].shape) == (NUM_CLASSES, *roi)


# ---------------------------------------------------------------------------
# Eval transforms: nessun patch sampling, volume intero
# ---------------------------------------------------------------------------


def test_eval_transform_returns_full_volume_no_cropping() -> None:
    data_root = _skip_if_no_real_data()
    dicts = build_subject_dicts(os.path.join(data_root, "val"))[:1]

    tf = build_eval_transforms(num_classes=NUM_CLASSES, with_dtm=False)
    ds = Dataset(data=dicts, transform=tf)
    sample = ds[0]

    # Nessun patch sampling: la shape spaziale deve essere quella originale
    # del volume BraTS-PEDs (240, 240, 155), non la ROI di training.
    assert tuple(sample["image"].shape) == (len(MODALITIES), 240, 240, 155)
    assert tuple(sample["label"].shape) == (1, 240, 240, 155)


# ---------------------------------------------------------------------------
# build_dataloaders_3d end-to-end (gia' verificato manualmente al Punto 5,
# qui formalizzato come test pytest permanente)
# ---------------------------------------------------------------------------


def test_build_dataloaders_3d_end_to_end() -> None:
    data_root = _skip_if_no_real_data()

    train_loader, val_loader = build_dataloaders_3d(
        data_root=data_root, roi=(64, 64, 64), num_classes=NUM_CLASSES,
        batch_size=2, num_samples=1, num_workers=0, with_dtm=False, use_cache=False,
    )

    train_batch = next(iter(train_loader))
    assert train_batch["image"].shape == (2, len(MODALITIES), 64, 64, 64)
    assert train_batch["label"].shape == (2, 1, 64, 64, 64)

    val_batch = next(iter(val_loader))
    assert val_batch["image"].shape[0] == 1  # batch_size fisso a 1 in validazione
    assert val_batch["image"].shape[1] == len(MODALITIES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
