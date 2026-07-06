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

        # get_not_nans=True: oltre alla media per-classe, MONAI restituisce il
        # "support" (numero di volumi in cui quella classe NON era degenere,
        # cioe' non-NaN al livello raw). Serve per la semantica BraTS-standard
        # (Opzione A, decisione utente): al livello per-volume MONAI da' gia'
        # Dice=1 se pred e GT sono ENTRAMBE vuote per quella classe (concordano),
        # Dice=0 se solo una e' vuota, e NaN per HD95 quando la classe e' assente.
        # Il problema era solo nell'AGGREGAZIONE: mean_batch trasforma in 0.0
        # (non NaN) una classe MAI presente in tutto il set, inquinando le medie.
        # Col support possiamo riconoscere quel caso (support==0) e riportare la
        # classe come NaN (esclusa dalle medie), invece dello 0.0 fittizio.
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch",  # media sui volumi, per-classe (NaN-aware)
            get_not_nans=True,
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=percentile,
            distance_metric=distance_metric,
            reduction="mean_batch",
            get_not_nans=True,
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
        # HD95 SEMPRE su CPU: MONAI instrada l'erosione morfologica sottostante
        # (get_mask_edges) su cucim (GPU) quando l'input e' un tensore CUDA, e su
        # alcune immagini Colab la compilazione JIT di cucim/CuPy crasha
        # (incompatibilita' del dialetto C++ negli header vendorizzati). Spostare
        # su CPU forza MONAI sul path scipy, bypassando cucim del tutto — costo
        # trascurabile: l'HD95 e' gia' intrinsecamente CPU-bound (erosioni
        # morfologiche), non un'operazione che beneficia molto dalla GPU.
        self.hd95_metric(y_pred=preds_oh.cpu(), y=targets_oh.cpu())

    def aggregate_and_reset(self) -> Dict[str, float]:
        """Calcola le medie finali per-classe e resetta gli accumulatori.

        Semantica BraTS-standard (Opzione A): una sub-regione MAI presente in
        tutto il set valutato (support==0) e' riportata come NaN — "non
        applicabile" — ed e' ESCLUSA dalle medie foreground, invece di entrarci
        come 0.0 fittizio (che falserebbe verso il basso dice_mean_fg e verso
        lo zero-perfetto hd95_mean_fg). Le classi presenti in almeno un volume
        conservano la loro media NaN-aware calcolata da MONAI (che al livello
        per-volume vale gia' 1 quando pred e GT concordano nel vuoto, 0 quando
        solo una e' vuota).

        Returns:
            Dict con chiavi "dice_<classe>" e "hd95_<classe>" per ciascuna
            delle sub-regioni foreground (ET, NET, CC, ED), piu' "dice_mean_fg"
            e "hd95_mean_fg" (media sulle sole sub-regioni PRESENTI nel set).
        """
        # Con get_not_nans=True, aggregate() ritorna (valori_per_classe, support):
        # support[i] = numero di volumi in cui la classe i non era degenere.
        dice_raw, dice_support = self.dice_metric.aggregate()   # [C'], [C']
        hd95_raw, hd95_support = self.hd95_metric.aggregate()

        results: Dict[str, float] = {}
        for i, name in enumerate(self.class_names):
            # support==0 -> classe mai presente nel set: NaN (non 0.0 fittizio).
            if i < len(dice_raw) and dice_support[i].item() > 0:
                results[f"dice_{name}"] = dice_raw[i].item()
            else:
                results[f"dice_{name}"] = float("nan")
            if i < len(hd95_raw) and hd95_support[i].item() > 0:
                results[f"hd95_{name}"] = hd95_raw[i].item()
            else:
                results[f"hd95_{name}"] = float("nan")

        fg_names = FOREGROUND_CLASS_NAMES
        # Entrambe le medie foreground ora ignorano simmetricamente i NaN (classi
        # a support nullo), coerentemente con la semantica BraTS: si media solo
        # sulle sub-regioni realmente presenti nel set. Se NESSUNA lo e', NaN.
        dice_vals = [results[f"dice_{n}"] for n in fg_names]
        finite_dice = [v for v in dice_vals if v == v]  # v==v e' False solo per NaN
        results["dice_mean_fg"] = (
            sum(finite_dice) / len(finite_dice) if finite_dice else float("nan")
        )
        hd95_vals = [results[f"hd95_{n}"] for n in fg_names]
        finite_hd95 = [v for v in hd95_vals if v == v]
        results["hd95_mean_fg"] = (
            sum(finite_hd95) / len(finite_hd95) if finite_hd95 else float("nan")
        )

        self.dice_metric.reset()
        self.hd95_metric.reset()

        return results
