"""
src/transforms_3d.py
======================
Transform MONAI custom per la normalizzazione delle intensita' MRI in
BraTS-PEDs 3D.

ClipAndNormalizeNonZerod
--------------------------
Replica ESATTA (a meno della dimensionalita' spaziale) della normalizzazione
del vecchio progetto 2D (master/src/dataset.py::clip_outliers +
zscore_normalise), applicata qui a volumi 3D interi invece che a slice 2D.
Decisione dell'utente: NON usare `monai.transforms.NormalizeIntensityd`
standard (che pur avendo un'opzione `nonzero=True` non applica il clip degli
outlier al percentile 99.5), per mantenere comparabilita' 1:1 con la pipeline
2D e la robustezza clinica contro gli artefatti da scanner gia' validata li'.

Logica, applicata in modo indipendente (channel-wise) su ciascuna delle 4
modalita' MRI:
    1. Maschera del cervello: voxel > 0 (il background BraTS-PEDs e' zero-padded).
    2. Percentile 99.5 calcolato SOLO sui voxel della maschera (non sull'intero
       volume, altrimenti il padding a zero distorcerebbe il percentile).
    3. Clip dei voxel della maschera a quel percentile (rimuove gli outlier
       da artefatto scanner, preservando la scala originale sotto la soglia).
    4. Media e deviazione standard calcolate SUI VOXEL GIA' CLIPPATI (non sui
       voxel grezzi pre-clip), cosi' gli outlier tagliati non influenzano piu'
       la normalizzazione — stesso ordine di operazioni del vecchio progetto.
    5. Z-score: (voxel - media) / std, applicato SOLO alla maschera.
    6. Il background (fuori maschera) resta ESATTAMENTE 0 dopo la trasformazione
       (non un valore vicino a zero: proprio 0), cosi' la rete puo' imparare ad
       associare il valore 0 esatto al "non-tessuto", coerentemente con la
       convenzione del vecchio progetto.
"""

from __future__ import annotations

from typing import Hashable, Mapping

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class ClipAndNormalizeNonZerod(MapTransform):
    """Clip al percentile 99.5 + Z-score, calcolati sui soli voxel non-zero.

    Applicata canale per canale (ogni modalita' MRI normalizzata in modo
    indipendente dalle altre, esattamente come nel vecchio progetto 2D).

    Args:
        keys:        Chiave (tipicamente "image") del tensore multi-canale
                     da normalizzare. Atteso [C, *spatial] (formato MONAI
                     post-EnsureChannelFirstd), con C = numero di modalita'.
        percentile:  Percentile superiore di clipping (default 99.5, come nel
                     vecchio progetto).
        min_nonzero_voxels: Soglia minima di voxel non-zero per considerare un
                     canale "popolato"; sotto questa soglia il canale viene
                     azzerato interamente (volume degenere), esattamente come
                     il fallback in `zscore_normalise` del vecchio progetto
                     (`if n < 10: return out`).
        allow_missing_keys: Standard MONAI.
    """

    def __init__(
        self,
        keys: KeysCollection,
        percentile: float = 99.5,
        min_nonzero_voxels: int = 10,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.percentile = percentile
        self.min_nonzero_voxels = min_nonzero_voxels

    def _clip_and_normalize_channel(self, channel: torch.Tensor) -> torch.Tensor:
        """Applica clip-P99.5 + z-score a un singolo canale [*spatial]."""
        mask = channel > 0
        n_nonzero = int(mask.sum().item())

        out = torch.zeros_like(channel)
        if n_nonzero < self.min_nonzero_voxels:
            return out  # canale degenere: rimane tutto zero (fallback esplicito)

        nonzero_vals = channel[mask]

        # Percentile 99.5 SOLO sui voxel non-zero (torch.quantile vuole [0,1]).
        upper = torch.quantile(nonzero_vals, self.percentile / 100.0)
        clipped_vals = torch.clamp(nonzero_vals, min=0.0, max=upper.item())

        mu = clipped_vals.mean()
        sigma = clipped_vals.std()
        if sigma.item() < 1e-8:
            return out  # varianza degenere: rimane tutto zero

        normalized_vals = (clipped_vals - mu) / sigma
        out[mask] = normalized_vals
        return out

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            volume = d[key]  # [C, *spatial]
            channels = [
                self._clip_and_normalize_channel(volume[c]) for c in range(volume.shape[0])
            ]
            d[key] = torch.stack(channels, dim=0)
        return d
