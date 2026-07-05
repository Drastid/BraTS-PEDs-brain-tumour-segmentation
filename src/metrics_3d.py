"""
src/metrics_3d.py
===================
Metriche di validazione/test 3D per BraTS-PEDs, via monai.metrics.

Requisito esplicito dell'utente (roadmap, punto 3 della direttiva): oltre alla
Dice, il loop di validazione (train_3d.py) e lo script di test devono
integrare monai.metrics.HausdorffDistanceMetric(percentile=95), calcolata e
loggata SEPARATAMENTE per le 4 sub-regioni pediatriche (ET, NET, CC, ED) — non
un unico valore aggregato.

Design
------
SegmentationMetrics3D incapsula sia DiceMetric sia HausdorffDistanceMetric di
MONAI in un'unica classe con interfaccia comune:

    metrics = SegmentationMetrics3D(num_classes=5)
    metrics.update(logits, targets)         # per batch, durante la validazione
    ...
    results = metrics.aggregate_and_reset() # a fine epoca/valutazione
    # results = {"dice_ET": ..., "dice_NET": ..., ..., "hd95_ET": ..., ...}

Entrambe le metriche MONAI si aspettano input one-hot [B,C,*spatial] sia per
la predizione (argmax poi one-hot) sia per il target (one-hot dell'indice di
classe) — la classe gestisce questa conversione internamente cosi' il
chiamante lavora sempre con logits/indici di classe grezzi, come nel resto
della pipeline.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# Nomi delle 4 sub-regioni tumorali pediatriche (esclude il background,
# indice 0) — coerente con src/constants.py::CLASS_NAMES.
FOREGROUND_CLASS_NAMES: Sequence[str] = ("ET", "NET", "CC", "ED")


def _to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converte un tensore di indici di classe [B,*spatial] in one-hot [B,C,*spatial]."""
    one_hot = torch.nn.functional.one_hot(indices, num_classes=num_classes)
    ndim = one_hot.ndim
    perm = [0, ndim - 1] + list(range(1, ndim - 1))
    return one_hot.permute(*perm).float()


class SegmentationMetrics3D:
    """Accumulatore di Dice + HD95 (percentile 95), per-classe, su volumi 3D.

    Le 4 sub-regioni pediatriche (ET, NET, CC, ED) sono sempre riportate
    separatamente — mai come media aggregata unica — cosi' e' possibile
    diagnosticare se il modello fatica su una specifica sub-regione (es. CC,
    la piu' rara).

    Args:
        num_classes:     Numero di classi totali, background incluso (5).
        include_background: Se includere la classe 0 nel calcolo di Dice/HD95.
                            Default False: il background raggiunge quasi
                            sempre Dice~1 e non e' informativo; le metriche
                            cliniche BraTS si riportano solo su foreground.
        percentile:      Percentile della Hausdorff Distance (95, come da
                         standard BraTS/richiesta esplicita dell'utente).
        distance_metric: Metrica di distanza sottostante ("euclidean", come
                         nel vecchio progetto via medpy).
    """

    def __init__(
        self,
        num_classes: int = 5,
        include_background: bool = False,
        percentile: float = 95.0,
        distance_metric: str = "euclidean",
    ) -> None:
        self.num_classes = num_classes
        self.include_background = include_background
        self.class_names = FOREGROUND_CLASS_NAMES if not include_background else (
            ("background",) + tuple(FOREGROUND_CLASS_NAMES)
        )

        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="none",  # nessuna riduzione: manteniamo il dettaglio per-classe
            get_not_nans=False,
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=percentile,
            distance_metric=distance_metric,
            reduction="none",
            get_not_nans=False,
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumula le metriche per un batch (o singolo volume).

        Args:
            logits:  Float tensor [B, C, *spatial] (raw, pre-softmax) OPPURE
                     gia' le predizioni one-hot/argmax — qui si assume siano
                     logits e si applica argmax internamente.
            targets: Long tensor [B, *spatial] (indici di classe, no canale).
        """
        preds_idx = logits.argmax(dim=1)  # [B, *spatial]
        preds_oh = _to_one_hot(preds_idx, self.num_classes)
        targets_oh = _to_one_hot(targets, self.num_classes)

        self.dice_metric(y_pred=preds_oh, y=targets_oh)
        self.hd95_metric(y_pred=preds_oh, y=targets_oh)

    def aggregate_and_reset(self) -> Dict[str, float]:
        """Calcola le medie finali per-classe e resetta gli accumulatori.

        Returns:
            Dict con chiavi "dice_<classe>" e "hd95_<classe>" per ciascuna
            delle sub-regioni foreground (ET, NET, CC, ED), piu' "dice_mean_fg"
            e "hd95_mean_fg" (media sulle 4 sub-regioni, per un riepilogo
            rapido oltre al dettaglio per-classe richiesto).
        """
        dice_raw = self.dice_metric.aggregate(reduction="mean_batch")  # [C'] (C'=4 se no bg)
        hd95_raw = self.hd95_metric.aggregate(reduction="mean_batch")

        results: Dict[str, float] = {}
        for i, name in enumerate(self.class_names):
            dice_val = dice_raw[i].item() if i < len(dice_raw) else float("nan")
            hd95_val = hd95_raw[i].item() if i < len(hd95_raw) else float("nan")
            results[f"dice_{name}"] = dice_val
            results[f"hd95_{name}"] = hd95_val

        fg_names = FOREGROUND_CLASS_NAMES
        results["dice_mean_fg"] = sum(results[f"dice_{n}"] for n in fg_names) / len(fg_names)
        # HD95 puo' essere NaN per classi assenti in un dato batch/volume:
        # la media va calcolata ignorando i NaN, altrimenti un solo NaN
        # propaga a NaN l'intero riepilogo aggregato.
        hd95_vals = [results[f"hd95_{n}"] for n in fg_names]
        finite_hd95 = [v for v in hd95_vals if v == v]  # v==v e' False solo per NaN
        results["hd95_mean_fg"] = (
            sum(finite_hd95) / len(finite_hd95) if finite_hd95 else float("nan")
        )

        self.dice_metric.reset()
        self.hd95_metric.reset()

        return results
