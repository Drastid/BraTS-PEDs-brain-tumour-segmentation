"""
src/models3d.py
=================
Architetture 3D volumetriche per BraTS-PEDs, via MONAI (roadmap §3).

Triade di modelli (rimpiazzano la triade 2D del vecchio progetto):

    2D (vecchio, master/)          3D (qui)
    ----------------------------   -----------------------------------
    U-Net (smp.Unet, ResNet34)     DynUNet   (stile nnU-Net)
    FPN   (smp.FPN,  ResNet34)     SegResNet (Myronenko 2018, vincitore BraTS)
    SegFormer (nvidia/mit-b1)      SwinUNETR (transformer 3D SOTA, SSL-pretrained)

FPN non ha equivalente 3D drop-in in MONAI (la FPN nasce per detection 2D).
Si rimpiazza con SegResNet, che realizza la stessa idea di aggregazione
multi-scala in ambito volumetrico ed e' l'unico dei tre con un bundle BraTS
pronto nel MONAI Model Zoo (si veda §3.2 di road_3D.md per la discussione
completa delle alternative).

Le architetture sono costruite SEMPRE con la testa a 5 classi pediatriche
(background, ET, NET, CC, ED — src/constants.py). I pesi pre-addestrati
pubblici (adulti, 3 canali TC/WT/ET o simili) si caricano DOPO con
`load_pretrained_3d`, che salta i tensori incompatibili per shape — tipicamente
l'ultimo layer di output — cosi' la head a 5 classi resta inizializzata da
zero mentre il resto del backbone eredita i pesi pre-addestrati.

Vincoli spaziali della ROI di training/inference (roadmap §3.3):
    - SwinUNETR: ogni dimensione della patch deve essere divisibile per 32
      (patch_size=2 di default => 2**5). Es. validi: 96, 128, 160, 192.
    - SegResNet / DynUNet (con gli stride di default qui sotto, 4 livelli di
      downsampling stride-2): patch divisibile per 16. 128 soddisfa entrambi
      i vincoli contemporaneamente.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, SegResNet, SwinUNETR

# Nomi degli slot mantenuti identici al vecchio progetto 2D (unet/fpn/segformer)
# per non rompere la convenzione di naming di checkpoint/run_pipeline. Ciò che
# restituiscono e' pero' DynUNet / SegResNet / SwinUNETR (si veda il docstring
# del modulo). Se preferisci nomi parlanti, valuta di rinominarli in un punto
# successivo e aggiornare i riferimenti in run_pipeline_3d.
ARCH_NAMES = ("unet", "fpn", "segformer")


def build_model_3d(
    arch: str,
    in_channels: int = 4,
    num_classes: int = 5,
    roi: Sequence[int] = (128, 128, 128),
) -> nn.Module:
    """Costruisce una delle tre architetture 3D con testa a `num_classes` canali.

    Args:
        arch:        Una tra "unet" (-> DynUNet), "fpn" (-> SegResNet, ex-FPN),
                     "segformer" (-> SwinUNETR, ex-SegFormer).
        in_channels: Numero di modalita' MRI in input (default 4: t1c,t1n,t2f,t2w).
        num_classes: Numero di classi di output (default 5: BG,ET,NET,CC,ED).
        roi:         Dimensione della patch 3D attesa in training/inference
                     (usata solo per DynUNet, che deriva `strides` dal numero
                     di livelli richiesti; SegResNet e SwinUNETR non ne hanno
                     bisogno alla costruzione, solo per il vincolo di
                     divisibilita' documentato nel modulo).

    Returns:
        Un nn.Module MONAI pronto per il forward su tensori
        [B, in_channels, *roi].

    Raises:
        ValueError: se `arch` non e' uno dei tre nomi attesi.
    """
    if arch == "unet":
        # DynUNet stile nnU-Net: 5 livelli (1 stem + 4 downsampling stride-2)
        # => richiede input divisibile per 16 su ogni asse spaziale.
        return DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            deep_supervision=False,
        )
    if arch == "fpn":
        # SegResNet — rimpiazza la FPN 2D (vedi docstring del modulo / road_3D.md §3.2)
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    if arch == "segformer":
        # SwinUNETR — rimpiazza SegFormer 2D; vincolo: patch divisibile per 32.
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
        )
    raise ValueError(
        f"arch sconosciuta: {arch!r} (attese: {', '.join(ARCH_NAMES)})"
    )


# ---------------------------------------------------------------------------
# Caricamento pesi pre-addestrati (agnostico rispetto alla fonte)
# ---------------------------------------------------------------------------


def load_pretrained_3d(
    model: nn.Module,
    ckpt_path: str,
    drop_prefixes: Iterable[str] = (),
    state_dict_key: Optional[str] = "state_dict",
    strip_module_prefix: bool = True,
    map_location: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Carica pesi pre-addestrati in un modello 3D, saltando i tensori incompatibili.

    Funzione generica e agnostica rispetto alla fonte del checkpoint: funziona
    con QUALSIASI file .pt/.pth, che provenga da un bundle MONAI Model Zoo, da
    pesi SSL NVIDIA, da un checkpoint HuggingFace, o da un training precedente
    di questo stesso progetto. Non assume nulla sulla architettura di origine:
    confronta chiave per chiave lo state_dict caricato con quello del modello
    corrente e tiene solo i tensori il cui NOME e SHAPE combaciano esattamente.

    Questo e' il meccanismo che rende sicuro il transfer learning quando la
    head di output ha un numero di classi diverso (es. checkpoint pre-addestrati
    a 3 canali TC/WT/ET adulti, mentre qui servono 5 classi pediatriche): la
    head viene automaticamente esclusa perche' la sua shape non combacia, e resta
    quindi inizializzata da zero (init di default del costruttore MONAI),
    mentre il resto del backbone eredita i pesi.

    Args:
        model:          Modello MONAI gia' costruito (es. via build_model_3d),
                        con la testa a `num_classes` GIA' della dimensione
                        desiderata (5 classi pediatriche).
        ckpt_path:      Path al file checkpoint (.pt/.pth).
        drop_prefixes:  Prefissi di nomi di parametro da escludere sempre,
                        anche se la shape combaciasse per caso (utile per
                        forzare il re-training di uno specifico blocco, es.
                        la head finale di un bundle con lo stesso num_classes
                        per coincidenza). Default: nessuno (si affida solo al
                        confronto di shape).
        state_dict_key: Chiave sotto cui il checkpoint annida lo state_dict
                        effettivo (tipico dei bundle: {"state_dict": {...}} o
                        {"model": {...}}). Se None, assume che il file caricato
                        SIA gia' direttamente lo state_dict. Se la chiave data
                        non esiste, si tenta un fallback automatico su "model"
                        e poi si assume il dict grezzo.
        strip_module_prefix: Se True, rimuove un eventuale prefisso "module."
                        (tipico di checkpoint salvati da nn.DataParallel /
                        DistributedDataParallel) prima del confronto.
        map_location:   Device di caricamento del checkpoint (default "cpu",
                        sicuro sia su macchine senza GPU sia come primo step
                        prima di spostare il modello su CUDA).
        verbose:        Se True, stampa un riepilogo di quanti tensori sono
                        stati caricati/saltati.

    Returns:
        Lo stesso `model` in input, con i pesi compatibili caricati in-place
        (`strict=False`: i tensori mancanti — tipicamente la head — restano
        con l'inizializzazione originale del costruttore).
    """
    raw = torch.load(ckpt_path, map_location=map_location)

    sd = raw
    if state_dict_key is not None and isinstance(raw, dict) and state_dict_key in raw:
        sd = raw[state_dict_key]
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    # altrimenti: raw e' gia' lo state_dict grezzo

    if strip_module_prefix:
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    drop_prefixes = tuple(drop_prefixes)
    model_sd = model.state_dict()

    kept = {}
    skipped_shape = []
    skipped_dropped = []
    for k, v in sd.items():
        if any(k.startswith(p) for p in drop_prefixes):
            skipped_dropped.append(k)
            continue
        if k not in model_sd:
            continue
        if v.shape != model_sd[k].shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        kept[k] = v

    missing, unexpected = model.load_state_dict(kept, strict=False)

    if verbose:
        print(
            f"[load_pretrained_3d] {ckpt_path}\n"
            f"  tensori caricati       : {len(kept)}/{len(model_sd)}\n"
            f"  saltati (shape mismatch): {len(skipped_shape)}\n"
            f"  saltati (drop_prefixes) : {len(skipped_dropped)}\n"
            f"  parametri del modello rimasti non inizializzati dal ckpt: {len(missing)}"
        )
        if skipped_shape:
            print("  --- shape mismatch (tipicamente la head di output) ---")
            for name, ckpt_shape, model_shape in skipped_shape[:10]:
                print(f"    {name}: ckpt{ckpt_shape} vs model{model_shape}")
            if len(skipped_shape) > 10:
                print(f"    ... e altri {len(skipped_shape) - 10}")

    return model
