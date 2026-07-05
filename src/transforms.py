"""
src/transforms.py
===================
Transform MONAI custom per BraTS-PEDs 3D.

ComputeDistanceMapd
--------------------
Calcola la Distance Transform Map (DTM) con segno, per-classe, richiesta dalla
Generalized Surface Loss (Celaya et al., si veda src/losses_3d.py). E' la
generalizzazione 3D di `compute_signed_dtm` dal vecchio progetto 2D
(master/src/dataset.py), con una differenza di design cruciale (roadmap §4.1
e decisione dell'utente):

    La DTM viene calcolata SOLO DOPO il patch sampling (es. dopo
    RandCropByPosNegLabeld), MAI sul volume intero. Motivazioni:
      1. Costo: scipy.ndimage.distance_transform_edt su un volume intero
         240x240x155 x 5 classi sarebbe troppo lento per essere ricalcolato
         ad ogni epoca; su una patch 128^3 x 5 classi costa pochi millisecondi.
      2. La DTM e' sulla ground truth, che e' una costante (non un tensore
         differenziabile) — non serve alcuna implementazione GPU/PyTorch:
         un'approssimazione GPU-friendly aggiungerebbe complessita' e
         imprecisione senza alcun beneficio, dato che il costo si nasconde
         gia' dietro l'asincronia del DataLoader (i worker CPU calcolano la
         DTM della prossima patch mentre la GPU processa il forward/backward
         della patch corrente).
      3. Usare l'EDT esatta (scipy), non un'approssimazione, mantiene la
         fedelta' matematica al paper (Eq. 12, Fig. 2).

Convenzione di segno (identica al vecchio progetto, paper Fig. 2): positivo
fuori dall'oggetto, zero sul bordo, negativo dentro l'oggetto.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np
import scipy.ndimage
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


def compute_signed_dtm_nd(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """DTM con segno, per-classe, su una maschera N-dimensionale (2D o 3D).

    Generalizzazione N-D di `compute_signed_dtm` dal vecchio progetto 2D:
    la logica e' identica (scipy.ndimage.distance_transform_edt lavora gia'
    nativamente in qualunque numero di dimensioni), cambia solo la shape di
    input/output.

    Args:
        mask:        Array di etichette intere, shape spaziale qualunque
                     (es. [H,W] in 2D o [D,H,W] in 3D — nessun canale).
        num_classes: Numero di classi C.

    Returns:
        Float32 array [C, *mask.shape] — la DTM con segno per ciascuna classe.
        Classi assenti nella maschera producono un campo tutto-zero (nessun
        contributo a numeratore/denominatore della GSL — comportamento corretto).
    """
    edt = scipy.ndimage.distance_transform_edt
    dtm = np.zeros((num_classes, *mask.shape), dtype=np.float32)

    for k in range(num_classes):
        binary = mask == k
        if not binary.any():
            continue
        if binary.all():
            inside = edt(binary).astype(np.float32)
            dtm[k] = -inside
            continue
        outside = edt(~binary).astype(np.float32)
        inside = edt(binary).astype(np.float32)
        dtm[k] = outside - inside

    return dtm


class ComputeDistanceMapd(MapTransform):
    """Transform MONAI: aggiunge la DTM con segno al dizionario del batch.

    Da comporre DOPO il patch sampling (es. dopo RandCropByPosNegLabeld), cosi'
    la DTM e' calcolata sulla patch gia' croppata (tipicamente 128^3), non sul
    volume intero. Scritta come MapTransform "normale" (non Randomizable): gira
    sui worker CPU del DataLoader in parallelo al forward/backward GPU del
    batch precedente, quindi il suo costo si nasconde dietro l'asincronia del
    DataLoader e non rallenta il training sull'A100.

    Args:
        keys:          Chiave (tipicamente "label") della maschera da cui
                       calcolare la DTM. Atteso un singolo canale [1, *spatial]
                       (formato MONAI post-EnsureChannelFirstd).
        num_classes:   Numero di classi C (5 per BraTS-PEDs pediatrico).
        distance_key:  Chiave sotto cui salvare la DTM risultante nel
                       dizionario (default "distance_map"), cosi' il training
                       loop puo' accedervi con `batch["distance_map"]`.
        allow_missing_keys: Standard MONAI — se True, non solleva errore
                       quando `keys` manca nel dizionario in input.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_classes: int,
        distance_key: str = "distance_map",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.num_classes = num_classes
        self.distance_key = distance_key

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            # Atteso [1, D, H, W] (canale singolo, post EnsureChannelFirstd).
            # Rimuove il canale per il calcolo, poi lo re-inserisce nell'output.
            label_np = label.squeeze(0).detach().cpu().numpy().astype(np.int16) \
                if isinstance(label, torch.Tensor) else np.asarray(label).squeeze(0).astype(np.int16)

            dtm = compute_signed_dtm_nd(label_np, num_classes=self.num_classes)  # [C, D, H, W]
            d[self.distance_key] = torch.from_numpy(dtm).float()
        return d
