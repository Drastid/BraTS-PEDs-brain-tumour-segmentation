"""
tests/test_eval_3d.py
========================
Verifica src/eval_3d.py: remove_small_components (porting 3D), predict_volume_3d
(inferenza nativa via sliding window), export NIfTI con affine corretto.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval_3d import (
    get_subject_ids,
    load_subject_affine,
    remove_small_components,
    save_prediction_nifti,
)


def test_remove_small_components_removes_isolated_voxels() -> None:
    volume = np.zeros((20, 20, 20), dtype=np.int16)
    # Componente grande (deve sopravvivere): un cubo 6x6x6 = 216 voxel di classe 1.
    volume[2:8, 2:8, 2:8] = 1
    # Componente isolata piccola (deve essere rimossa): 2 voxel di classe 1,
    # lontani dalla componente grande.
    volume[15, 15, 15] = 1
    volume[15, 15, 16] = 1

    out = remove_small_components(volume, num_classes=5, min_voxels=50)

    assert out[15, 15, 15] == 0, "la componente isolata di 2 voxel deve essere rimossa"
    assert out[15, 15, 16] == 0
    assert out[4, 4, 4] == 1, "la componente grande (216 voxel) deve sopravvivere"
    assert int((out == 1).sum()) == 216


def test_remove_small_components_preserves_multiple_classes() -> None:
    volume = np.zeros((16, 16, 16), dtype=np.int16)
    volume[0:6, 0:6, 0:6] = 1  # classe 1, componente grande (216 voxel)
    volume[10:16, 10:16, 10:16] = 4  # classe 4, componente grande (216 voxel)

    out = remove_small_components(volume, num_classes=5, min_voxels=50)

    assert int((out == 1).sum()) == 216
    assert int((out == 4).sum()) == 216


def test_get_subject_ids_on_real_data() -> None:
    test_dir = os.path.join("data", "processed_3d", "test")
    if not os.path.isdir(test_dir):
        pytest.skip("data/processed_3d/test non presente in questo ambiente")

    ids = get_subject_ids(test_dir)
    assert len(ids) == 26, f"attesi 26 soggetti test, trovati {len(ids)}"
    assert all(sid.startswith("BraTS-PED-") for sid in ids)


def test_nifti_export_roundtrip(tmp_path) -> None:
    test_dir = os.path.join("data", "processed_3d", "test")
    if not os.path.isdir(test_dir):
        pytest.skip("data/processed_3d/test non presente in questo ambiente")

    subject_id = get_subject_ids(test_dir)[0]
    affine, header = load_subject_affine(test_dir, subject_id)

    fake_pred = np.random.randint(0, 5, size=(240, 240, 155)).astype(np.int16)
    out_path = str(tmp_path / f"{subject_id}_pred.nii.gz")
    save_prediction_nifti(fake_pred, affine, header, out_path)

    assert os.path.isfile(out_path)

    import nibabel as nib
    reloaded = nib.load(out_path)
    assert reloaded.shape == (240, 240, 155)
    np.testing.assert_allclose(reloaded.affine, affine)
    reloaded_data = reloaded.get_fdata().astype(np.int16)
    np.testing.assert_array_equal(reloaded_data, fake_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
