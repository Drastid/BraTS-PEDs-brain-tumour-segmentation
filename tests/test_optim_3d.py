"""
tests/test_optim_3d.py
=========================
Verifica src/optim_3d.py: split backbone/head, LR differenziato, freeze
opzionale del backbone — per tutte e tre le architetture 3D.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models3d import ARCH_NAMES, build_model_3d
from src.optim_3d import build_optimizer_3d, set_backbone_trainable, split_backbone_head_params


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_split_covers_all_params_disjointly(arch: str) -> None:
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))
    all_params = list(model.parameters())
    backbone, head = split_backbone_head_params(model, arch)

    assert len(backbone) + len(head) == len(all_params)
    backbone_ids = {id(p) for p in backbone}
    head_ids = {id(p) for p in head}
    assert backbone_ids.isdisjoint(head_ids)
    assert len(head) > 0, f"[{arch}] la head non deve avere zero parametri"
    assert len(backbone) > 0, f"[{arch}] il backbone non deve avere zero parametri"


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_optimizer_has_differentiated_lr(arch: str) -> None:
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))
    base_lr = 1e-3
    opt = build_optimizer_3d(model, arch, base_lr=base_lr, backbone_lr_mult=0.1)

    assert len(opt.param_groups) == 2
    backbone_group, head_group = opt.param_groups
    assert backbone_group["lr"] == pytest.approx(base_lr * 0.1)
    assert head_group["lr"] == pytest.approx(base_lr)


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_freeze_unfreeze_backbone(arch: str) -> None:
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))
    backbone, head = split_backbone_head_params(model, arch)

    set_backbone_trainable(model, arch, trainable=False)
    assert all(not p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in head), "la head deve restare sempre allenabile"

    set_backbone_trainable(model, arch, trainable=True)
    assert all(p.requires_grad for p in backbone)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
