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


# ---------------------------------------------------------------------------
# extra_head_param_names — parametri non pre-addestrati FUORI dalla head
# strutturale (es. patch_embed di SwinUNETR con model_swinvit.pt, escluso da
# load_pretrained_3d per mismatch canali 1 vs 4). Devono ricevere lo stesso
# trattamento della head: LR pieno, mai congelati durante il warm-up.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_extra_head_param_names_moves_params_into_head_group(arch: str) -> None:
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))

    # Prendo un parametro qualunque del backbone "di base" (non della head
    # strutturale) e lo forzo come extra head param, simulando un layer
    # scartato per shape mismatch fuori dalla head (es. patch_embed).
    backbone_before, head_before = split_backbone_head_params(model, arch)
    assert len(backbone_before) > 1, f"[{arch}] serve almeno 2 parametri nel backbone per questo test"

    name_to_param = dict(model.named_parameters())
    param_to_name = {id(p): n for n, p in name_to_param.items()}
    extra_name = param_to_name[id(backbone_before[0])]

    backbone_after, head_after = split_backbone_head_params(
        model, arch, extra_head_param_names=[extra_name],
    )

    assert id(backbone_before[0]) not in {id(p) for p in backbone_after}
    assert id(backbone_before[0]) in {id(p) for p in head_after}
    assert len(backbone_after) + len(head_after) == len(backbone_before) + len(head_before)


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_extra_head_param_names_gets_full_lr(arch: str) -> None:
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))
    base_lr = 1e-3

    backbone_before, _ = split_backbone_head_params(model, arch)
    name_to_param = dict(model.named_parameters())
    param_to_name = {id(p): n for n, p in name_to_param.items()}
    extra_name = param_to_name[id(backbone_before[0])]

    opt = build_optimizer_3d(
        model, arch, base_lr=base_lr, backbone_lr_mult=0.1,
        extra_head_param_names=[extra_name],
    )
    backbone_group, head_group = opt.param_groups

    assert id(backbone_before[0]) not in {id(p) for p in backbone_group["params"]}
    assert id(backbone_before[0]) in {id(p) for p in head_group["params"]}
    assert head_group["lr"] == pytest.approx(base_lr)


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_extra_head_param_names_survives_freeze(arch: str) -> None:
    """Il parametro forzato come 'head' non deve MAI essere congelato dal
    warm-up, anche se strutturalmente appartiene al backbone."""
    model = build_model_3d(arch, num_classes=5, roi=(64, 64, 64))

    backbone_before, _ = split_backbone_head_params(model, arch)
    name_to_param = dict(model.named_parameters())
    param_to_name = {id(p): n for n, p in name_to_param.items()}
    extra_name = param_to_name[id(backbone_before[0])]
    extra_param = backbone_before[0]

    set_backbone_trainable(model, arch, trainable=False, extra_head_param_names=[extra_name])

    assert extra_param.requires_grad, (
        f"[{arch}] parametro extra_head_param_names non deve essere congelato dal warm-up"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
