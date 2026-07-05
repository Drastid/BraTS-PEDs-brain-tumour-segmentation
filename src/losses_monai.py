"""
src/losses_monai.py
======================
Pipeline di loss "sicura" (default) per il training baseline: wrapper sottile
attorno a monai.losses.DiceFocalLoss, gia' validata su BraTS da MONAI stesso
(roadmap §4, Strada A). Usata come default in run_pipeline_3d.py quando
--loss=dice_focal; l'alternativa --loss=gsl usa invece src/losses_3d.py
(DiceFocalGSLLoss, porting N-D delle loss custom del progetto).

MONAI si aspetta il target con canale esplicito [B, 1, *spatial], mentre il
resto della pipeline (dataset, metriche) segue la convenzione [B, *spatial]
(indici di classe, nessun canale) — coerente con src/losses_3d.py e col
vecchio progetto 2D. build_dice_focal_loss() incapsula questo adattamento
cosi' il training loop puo' passare i targets nella forma "naturale".
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss


class DiceFocalLossWrapper(nn.Module):
    """Adatta monai.losses.DiceFocalLoss alla convenzione [B, *spatial] (no
    canale) usata dal resto della pipeline, invece della convenzione nativa
    MONAI [B, 1, *spatial].

    Args:
        num_classes:        Numero di classi (5 per BraTS-PEDs pediatrico).
        include_background: Se False, la classe 0 e' esclusa (== ignore_background
                            del vecchio progetto).
        gamma:              Parametro di focusing della componente Focal.
        weight:             Pesi per-classe (inverse-frequency) su TUTTE e
                            num_classes classi (indice 0 = background
                            incluso), a prescindere da `include_background`.
                            Se `include_background=False`, il peso della
                            classe 0 viene automaticamente scartato prima di
                            passarlo a MONAI: la sua API richiede infatti che
                            `weight` abbia lunghezza pari al numero di classi
                            EFFETTIVAMENTE mediate (num_classes-1 in quel
                            caso), non num_classes — un dettaglio facile da
                            mancare che solleverebbe altrimenti un
                            ValueError a runtime.
        lambda_dice:        Peso del termine Dice nella somma pesata.
        lambda_focal:       Peso del termine Focal nella somma pesata.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        gamma: float = 2.0,
        weight: Optional[Sequence[float]] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        weight_t: Optional[torch.Tensor] = None
        if weight is not None:
            weight_t = torch.as_tensor(weight, dtype=torch.float32)
            if not include_background:
                weight_t = weight_t[1:]  # MONAI vuole solo i pesi foreground
        self.loss = DiceFocalLoss(
            include_background=include_background,
            softmax=True,
            to_onehot_y=True,
            gamma=gamma,
            weight=weight_t,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcola la loss.

        Args:
            logits:  Float tensor [B, C, *spatial] (raw, pre-softmax).
            targets: Long tensor  [B, *spatial]    (indici di classe, NO canale).

        Returns:
            Scalar loss tensor.
        """
        targets_ch = targets.unsqueeze(1)  # [B, *spatial] -> [B, 1, *spatial]
        return self.loss(logits, targets_ch)


def build_dice_focal_loss(
    num_classes: int,
    class_weights: Optional[Sequence[float]] = None,
    gamma: float = 2.0,
    ignore_background: bool = True,
) -> DiceFocalLossWrapper:
    """Factory di convenienza, con firma allineata a CombinedLoss/DiceFocalGSLLoss
    (src/losses_3d.py) per rendere i due rami --loss intercambiabili nel
    training loop.
    """
    return DiceFocalLossWrapper(
        num_classes=num_classes,
        include_background=not ignore_background,
        gamma=gamma,
        weight=class_weights,
    )
