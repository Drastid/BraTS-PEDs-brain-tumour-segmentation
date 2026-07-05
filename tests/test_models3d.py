"""
tests/test_models3d.py
========================
Smoke test per src/models3d.py (roadmap §9): verifica che le tre architetture
3D costruiscano correttamente e producano output della shape attesa su un
input fittizio [B, 4, 128, 128, 128] -> [B, 5, 128, 128, 128].

Esecuzione locale su CPU (no CUDA) — solo verifica strutturale, non prestazionale.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models3d import ARCH_NAMES, build_model_3d

ROI = (128, 128, 128)
IN_CHANNELS = 4
NUM_CLASSES = 5


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_build_and_forward(arch: str) -> None:
    model = build_model_3d(arch, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, roi=ROI)
    model.eval()

    x = torch.randn(1, IN_CHANNELS, *ROI)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, NUM_CLASSES, *ROI), (
        f"[{arch}] output shape {tuple(y.shape)} != atteso "
        f"{(1, NUM_CLASSES, *ROI)}"
    )


def test_invalid_arch_raises() -> None:
    with pytest.raises(ValueError):
        build_model_3d("not-a-real-arch")


if __name__ == "__main__":
    for arch in ARCH_NAMES:
        print(f"--- {arch} ---")
        test_build_and_forward(arch)
        print(f"  OK: output shape corretta.")
    test_invalid_arch_raises()
    print("Tutti i test superati.")
