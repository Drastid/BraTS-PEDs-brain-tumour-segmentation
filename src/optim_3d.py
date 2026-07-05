"""
src/optim_3d.py
=================
Ottimizzatore a LR differenziato per il fine-tuning dei modelli 3D
pre-addestrati (roadmap §5.3, decisione utente: implementarlo gia' ora invece
di partire con un LR singolo uniforme).

Motivazione (road_3D.md §3.4/§5.3): i tre modelli 3D partono da pesi
pre-addestrati (transfer learning), non da zero. Il vecchio progetto 2D usava
un two-phase schedule legato all'encoder ImageNet (freeze poi unfreeze). Qui
non c'e' un ImageNet 3D standard, ma la stessa idea si applica al backbone
pre-addestrato (SegResNet bundle BraTS, SwinUNETR SSL): LR piu' basso sul
backbone (che ha gia' feature sensate) e LR pieno sulla head di output (che è
sempre inizializzata da zero a 5 classi pediatriche, si veda
src/models3d.py::load_pretrained_3d).

set_backbone_trainable() e' l'analogo 3D di set_encoder_trainable() del
vecchio progetto: permette un warm-up opzionale a backbone congelato (utile
soprattutto per SwinUNETR, i cui pesi SSL sono piu' delicati da rovinare nelle
prime epoche).

Identificazione backbone vs head
----------------------------------
Le tre architetture MONAI hanno nomi di modulo diversi per la head di output:
    - DynUNet:    ultimo layer e' `output_block` (o `deep_supervision_heads`
                  se deep_supervision=True, non usato qui).
    - SegResNet:  ultimo layer e' `conv_final` (visto anche nel test di
                  src/models3d.py::load_pretrained_3d).
    - SwinUNETR:  ultimo layer e' `out` (il conv finale dopo il decoder UNet).
Questa mappa e' incapsulata in _HEAD_MODULE_NAMES cosi' il chiamante non deve
conoscere i dettagli interni di ciascuna architettura MONAI.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

# Nome del modulo "head" (ultimo layer di output) per ciascuna architettura,
# nella stessa convenzione di naming di arch usata da src/models3d.py.
_HEAD_MODULE_NAMES: Dict[str, str] = {
    "unet": "output_block",  # DynUNet
    "fpn": "conv_final",  # SegResNet (ex-FPN)
    "segformer": "out",  # SwinUNETR (ex-SegFormer)
}


def _get_head_module(model: nn.Module, arch: str) -> nn.Module:
    """Ritorna il sotto-modulo 'head' di output per l'architettura data.

    Args:
        model: Modello costruito da src.models3d.build_model_3d.
        arch:  Uno tra "unet", "fpn", "segformer" (stessa convenzione di
               src.models3d.ARCH_NAMES).

    Returns:
        Il nn.Module della head.

    Raises:
        ValueError: se `arch` non e' riconosciuta.
        AttributeError: se il modello non espone l'attributo atteso (es. e'
            stata cambiata la versione di MONAI con nomi diversi).
    """
    if arch not in _HEAD_MODULE_NAMES:
        raise ValueError(f"arch sconosciuta: {arch!r} (attese: {list(_HEAD_MODULE_NAMES)})")
    attr_name = _HEAD_MODULE_NAMES[arch]
    if not hasattr(model, attr_name):
        raise AttributeError(
            f"Il modello per arch={arch!r} non espone l'attributo '{attr_name}'. "
            f"Verifica la versione di MONAI o l'architettura costruita."
        )
    return getattr(model, attr_name)


def split_backbone_head_params(
    model: nn.Module, arch: str, extra_head_param_names: Iterable[str] = (),
) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Divide i parametri del modello in (backbone, head) per LR differenziato.

    La head "strutturale" (modulo di output dell'architettura) e' identificata
    per IDENTITA' dei tensori (id(p)), non per nome stringa, cosi' la
    distinzione resta corretta indipendentemente da come i parametri sono
    enumerati da .parameters() (stessa tecnica gia' usata nel vecchio progetto
    2D per separare encoder/decoder — si veda
    master/run_pipeline.py::build_optim_sched).

    Args:
        model: Modello costruito da build_model_3d.
        arch:  "unet" | "fpn" | "segformer".
        extra_head_param_names: Nomi (convenzione model.state_dict()/
            model.named_parameters()) di parametri AGGIUNTIVI da trattare come
            head, anche se non appartengono al modulo di output. Pensato per i
            nomi restituiti da src.models3d.load_pretrained_3d come
            "unmatched_param_names": un layer come patch_embed, escluso dal
            caricamento per shape mismatch (es. checkpoint SSL NVIDIA a 1
            canale vs le 4 modalita' richieste qui), resta inizializzato a
            caso esattamente come la head — merita quindi lo stesso LR pieno,
            non il LR ridotto del backbone pre-addestrato. Default: nessuno
            (comportamento identico a prima per checkpoint dove solo la head
            "strutturale" e' esclusa, es. model_best_fold_0.pth).

    Returns:
        (backbone_params, head_params) — due liste disgiunte che coprono
        insieme TUTTI i parametri del modello.
    """
    head_module = _get_head_module(model, arch)
    head_ids = {id(p) for p in head_module.parameters()}

    extra_names = set(extra_head_param_names)
    if extra_names:
        for name, p in model.named_parameters():
            if name in extra_names:
                head_ids.add(id(p))

    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    return backbone_params, head_params


def set_backbone_trainable(
    model: nn.Module, arch: str, trainable: bool, extra_head_param_names: Iterable[str] = (),
) -> None:
    """Congela/scongela il backbone (tutto tranne la head) per un warm-up opzionale.

    Analogo 3D di set_encoder_trainable() dal vecchio progetto 2D. Utile per
    SwinUNETR: congelare temporaneamente il backbone SSL per le prime epoche
    permette alla head (inizializzata da zero) di "scaldarsi" senza che i
    gradienti iniziali, spesso rumorosi, rovinino i pesi pre-addestrati.

    Args:
        model:     Modello costruito da build_model_3d.
        arch:      "unet" | "fpn" | "segformer".
        trainable: True -> il backbone e' allenabile; False -> congelato
                   (requires_grad=False su tutti i parametri fuori dalla head).
        extra_head_param_names: Si veda split_backbone_head_params — questi
                   parametri sono sempre esclusi dal freeze (restano
                   allenabili) perche' inizializzati a caso, non pre-addestrati.
    """
    backbone_params, _ = split_backbone_head_params(model, arch, extra_head_param_names)
    for p in backbone_params:
        p.requires_grad = trainable

    state = "allenabile" if trainable else "congelato"
    n_params = sum(p.numel() for p in backbone_params)
    print(f"[set_backbone_trainable] Backbone ({arch}) ora {state} ({n_params:,} parametri).")


def build_optimizer_3d(
    model: nn.Module,
    arch: str,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
    weight_decay: float = 1e-4,
    extra_head_param_names: Iterable[str] = (),
) -> torch.optim.Optimizer:
    """AdamW con 2 param-group: backbone a LR ridotto, head nuova a LR pieno.

    Args:
        model:            Modello costruito da build_model_3d (con pesi
                          pre-addestrati gia' caricati via load_pretrained_3d,
                          se applicabile).
        arch:             "unet" | "fpn" | "segformer".
        base_lr:          LR di riferimento (applicato per intero alla head).
        backbone_lr_mult: Moltiplicatore per il LR del backbone
                          (backbone_lr = base_lr * backbone_lr_mult). Default
                          0.1, stesso valore di encoder_lr_mult nel vecchio
                          progetto 2D.
        weight_decay:      Weight decay AdamW (uguale per entrambi i gruppi).
        extra_head_param_names: Si veda split_backbone_head_params — nomi di
                          parametri non pre-addestrati (shape mismatch fuori
                          dalla head strutturale, es. patch_embed con
                          model_swinvit.pt) da trattare a LR pieno.

    Returns:
        torch.optim.AdamW con 2 param_group: [0]=backbone, [1]=head.
    """
    backbone_params, head_params = split_backbone_head_params(model, arch, extra_head_param_names)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )
