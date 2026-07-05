"""
src/losses_3d.py
==================
Porting N-dimensionale delle loss custom del vecchio progetto 2D
(master/src/losses.py), utilizzabili sia in 2D [B,C,H,W] sia in 3D
[B,C,D,H,W] senza modifiche — generalizzazione richiesta esplicitamente
dall'utente per conservare la logica scientifica gia' validata (paper Celaya
et al., Generalized Surface Loss) anche nella pipeline 3D.

Classi
------
DiceLoss             Soft multi-class Dice loss, N-D.
FocalLoss             Multi-class focal loss, N-D.
CombinedLoss          Dice + Focal pesati (baseline "sicura": vedi anche
                      src/losses_monai.py per l'equivalente MONAI usato come
                      pipeline di default).
GeneralizedSurfaceLoss  GSL (Eq. 12 del paper), N-D.
DiceFocalGSLLoss      Loss schedulata alpha(t)*DiceFocal + (1-alpha(t))*GSL
                      (Eq. 8 del paper), N-D.
AlphaScheduler        Schedule epoch-indicizzato per alpha(t) (invariato dal
                      vecchio progetto — e' gia' dimension-agnostico).
compute_global_class_weights  Pesi globali w_k (Eq. 13) — invariato.

Differenze rispetto al vecchio master/src/losses.py (2D-only)
----------------------------------------------------------------
- `F.one_hot(...).permute(0,3,1,2)` (fisso a 4 assi) diventa una permute
  generica che sposta l'ultimo asse (canale) in posizione 1 per QUALUNQUE
  rank del tensore (funziona sia per [B,H,W,C]->[B,C,H,W] sia per
  [B,D,H,W,C]->[B,C,D,H,W]).
- `.sum(dim=(1,2))` (fisso a H,W) diventa `.sum(dim=tuple(range(2, ndim)))`,
  che somma su tutte le dimensioni spaziali qualunque sia il numero di assi.
- In FocalLoss, `B,C,H,W = logits.shape` (spacchettamento fisso) diventa una
  gestione generica di forma/permute/reshape basata su `logits.ndim`.

La GSL (`compute_signed_dtm`) NON e' qui: quella parte (il calcolo della DTM)
vive in `src/transforms.py` come transform MONAI applicata dopo il patch
sampling (si veda quel modulo per il ragionamento su costo/dove-gira).
Questo modulo consuma la DTM gia' calcolata (tensore `dtm` in input al
forward di `GeneralizedSurfaceLoss`), esattamente come il vecchio progetto.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper N-D condivisi
# ---------------------------------------------------------------------------


def _one_hot_channel_first(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot di un tensore di indici di classe, con canale in posizione 1.

    Generalizza `F.one_hot(...).permute(0,3,1,2)` (fisso 2D) a qualunque rank:
    funziona sia per targets [B,H,W] -> one_hot [B,C,H,W] (2D), sia per
    targets [B,D,H,W] -> one_hot [B,C,D,H,W] (3D).

    Args:
        targets:     Long tensor [B, *spatial] con indici di classe.
        num_classes: Numero di classi C.

    Returns:
        Float tensor [B, C, *spatial].
    """
    one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, *spatial, C]
    # Sposta l'ultimo asse (C) in posizione 1: [B, *spatial, C] -> [B, C, *spatial]
    ndim = one_hot.ndim
    perm = [0, ndim - 1] + list(range(1, ndim - 1))
    return one_hot.permute(*perm).float()


def _spatial_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    """Ritorna gli indici delle dimensioni spaziali di un tensore [B,C,*spatial].

    Per [B,C,H,W] (2D) -> (2,3). Per [B,C,D,H,W] (3D) -> (2,3,4).
    """
    return tuple(range(2, tensor.ndim))


# ---------------------------------------------------------------------------
# Dice Loss (N-D)
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, N-dimensionale (2D o 3D senza modifiche).

    Identica in formula al vecchio master/src/losses.py::DiceLoss; l'unica
    differenza e' che le riduzioni sommano su TUTTE le dimensioni spaziali
    presenti nel tensore in input (2 per 2D, 3 per 3D), invece di assumere
    esattamente (H, W).

    Args:
        num_classes:       Numero di classi (5 per BraTS-PEDs pediatrico).
        ignore_background: Se True, la classe 0 e' esclusa dalla media.
        class_weights:      Pesi per-classe opzionali (L1-normalizzati
                            internamente sul sottoinsieme mediato).
        smooth:             Termine di Laplace-smoothing (default 1e-6).
    """

    def __init__(
        self,
        num_classes: int,
        ignore_background: bool = True,
        class_weights: Optional[Sequence[float]] = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            w = w / w.sum()
            self.register_buffer("class_weights", w)
        else:
            self.class_weights: Optional[torch.Tensor] = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcola la Dice loss.

        Args:
            logits:  Float tensor [B, C, *spatial] (raw, pre-softmax).
            targets: Long tensor  [B, *spatial]    (indici di classe).

        Returns:
            Scalar loss tensor.
        """
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot_channel_first(targets, self.num_classes)  # [B,C,*spatial]

        dims = _spatial_dims(probs)  # (2,3) in 2D, (2,3,4) in 3D
        start_c = 1 if self.ignore_background else 0
        classes = list(range(start_c, self.num_classes))

        dice_per_class = []
        for c in classes:
            p = probs[:, c]
            y = one_hot[:, c]
            intersection = (p * y).sum(dim=tuple(d - 1 for d in dims))  # [B] (canale gia' rimosso)
            cardinality = p.sum(dim=tuple(d - 1 for d in dims)) + y.sum(dim=tuple(d - 1 for d in dims))
            dice_c = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_per_class.append(dice_c.mean())

        dice_tensor = torch.stack(dice_per_class)

        if self.class_weights is not None:
            w = self.class_weights[start_c:].to(dice_tensor.device)
            w = w / w.sum()
            return (dice_tensor * w).sum()

        return dice_tensor.mean()


# ---------------------------------------------------------------------------
# Focal Loss (N-D)
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss, N-dimensionale (2D o 3D senza modifiche).

    Args:
        gamma:        Parametro di focusing gamma >= 0 (default 2.0).
        alpha:        Pesi per-classe opzionali (lunghezza C).
        ignore_index: Indice di classe da ignorare (default -100 = nessuno).
        reduction:    'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            a = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a)
        else:
            self.alpha: Optional[torch.Tensor] = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcola la focal loss.

        Args:
            logits:  Float tensor [B, C, *spatial] (raw, pre-softmax).
            targets: Long tensor  [B, *spatial]    (indici di classe).

        Returns:
            Scalar (o per-elemento) loss tensor.
        """
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, *spatial]
        probs = log_probs.exp()

        C = logits.shape[1]
        ndim = logits.ndim
        # [B, C, *spatial] -> [B, *spatial, C] -> [N, C], per qualunque rank.
        perm_to_last = [0] + list(range(2, ndim)) + [1]
        log_probs_flat = log_probs.permute(*perm_to_last).reshape(-1, C)
        probs_flat = probs.permute(*perm_to_last).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        log_pt = log_probs_flat.gather(1, targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)
        pt = probs_flat.gather(1, targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, targets_flat.clamp(min=0))
            focal_weight = focal_weight * at

        loss = -focal_weight * log_pt

        if self.ignore_index >= 0:
            valid = targets_flat != self.ignore_index
            loss = loss[valid]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Combined Loss (N-D)
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Somma pesata di DiceLoss + FocalLoss, N-dimensionale.

    Args:
        num_classes:       Numero di classi.
        dice_weight:       Peso del termine Dice (default 1.0).
        focal_weight:      Peso del termine Focal (default 1.0).
        gamma:             Parametro di focusing della Focal loss.
        ignore_background: Se True, esclude la classe 0 dalla media Dice.
        class_weights:      Pesi per-classe opzionali, applicati sia a Dice
                            che a Focal.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        ignore_background: bool = True,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights,
            smooth=1e-6,
        )
        self.focal_loss = FocalLoss(gamma=gamma, alpha=class_weights, reduction="mean")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcola la loss combinata.

        Returns:
            total, d_loss, f_loss (tutti scalari; d_loss/f_loss per logging).
        """
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        total = self.dice_weight * d_loss + self.focal_weight * f_loss
        return total, d_loss, f_loss


# ===========================================================================
# Generalized Surface Loss (GSL) — Celaya et al., arXiv:2302.03868 — N-D
# ===========================================================================


def compute_global_class_weights(voxel_counts: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    """Pre-calcola i pesi globali GSL w_k (Eq. 13). Identico al vecchio progetto
    (gia' dimension-agnostico: opera su conteggi scalari per classe, non su
    tensori spaziali).
    """
    counts = torch.as_tensor(voxel_counts, dtype=torch.float64)
    inv = 1.0 / (counts + eps)
    w = inv / inv.sum()
    return w.to(dtype=torch.float32)


class AlphaScheduler:
    """Schedule epoch-indicizzato per alpha(t) (Eq. 14-16). Invariato dal
    vecchio progetto: non dipende dalla dimensionalita' spaziale.
    """

    def __init__(self, schedule: str = "step", total_epochs: int = 30, step_length: int = 5) -> None:
        if schedule not in ("linear", "step", "cosine"):
            raise ValueError(
                f"Unknown alpha schedule {schedule!r}; expected 'linear', 'step', or 'cosine'."
            )
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}.")
        if step_length < 1:
            raise ValueError(f"step_length must be >= 1, got {step_length}.")
        self.schedule = schedule
        self.total_epochs = int(total_epochs)
        self.step_length = int(step_length)

    def __call__(self, epoch: int) -> float:
        import math

        t = int(epoch)
        T = self.total_epochs
        denom = max(T - 1, 1)

        if self.schedule == "linear":
            alpha = 1.0 - t / denom
        elif self.schedule == "step":
            h = self.step_length
            n_h = max((T - 1) // h, 1)
            alpha = 1.0 - (t // h) / n_h
        else:  # cosine
            alpha = 0.5 * (1.0 + math.cos(math.pi * t / denom))

        return float(min(1.0, max(0.0, alpha)))


class GeneralizedSurfaceLoss(nn.Module):
    """GSL — boundary-aware, bounded in [0, 1] (Eq. 12), N-dimensionale.

    Consuma una DTM gia' calcolata (si veda src/transforms.py::ComputeDistanceMapd),
    quindi opera su tensori [B, C, *spatial] generici — funziona identicamente
    in 2D [B,C,H,W] e 3D [B,C,D,H,W].

    Args:
        num_classes:        Numero di classi C.
        class_weights:       Pesi globali w_k (Eq. 13). Se None, uniformi.
        include_background: Se True, la classe 0 contribuisce alle somme GSL.
        eps:                Costante di stabilizzazione numerica.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Sequence[float]] = None,
        include_background: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() != num_classes:
                raise ValueError(
                    f"class_weights must have length num_classes={num_classes}, got {w.numel()}."
                )
        else:
            w = torch.ones(num_classes, dtype=torch.float32) / num_classes
        self.register_buffer("class_weights", w)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, dtm: torch.Tensor
    ) -> torch.Tensor:
        """Calcola la GSL.

        Args:
            logits:  Float tensor [B, C, *spatial] (raw, pre-softmax).
            targets: Long tensor  [B, *spatial]    (indici di classe).
            dtm:     Float tensor [B, C, *spatial] — DTM con segno della GT,
                     gia' allineata spazialmente a logits/targets (stessa
                     patch, gia' croppata da ComputeDistanceMapd).

        Returns:
            Scalar GSL in [0, 1].
        """
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot_channel_first(targets, self.num_classes)

        start_c = 0 if self.include_background else 1
        w = self.class_weights[start_c:].to(probs.device)
        D = dtm[:, start_c:]
        T = one_hot[:, start_c:]
        P = probs[:, start_c:]

        dims = tuple(range(0, D.ndim))
        # Somma su (batch, *spatial) — tutte le dimensioni tranne il canale (asse 1).
        sum_dims = (0,) + tuple(range(2, D.ndim))

        num_vox = (D * (1.0 - (T + P))) ** 2
        den_vox = D ** 2

        num_per_class = num_vox.sum(dim=sum_dims)
        den_per_class = den_vox.sum(dim=sum_dims)

        numerator = (w * num_per_class).sum()
        denominator = (w * den_per_class).sum()

        return 1.0 - numerator / (denominator + self.eps)


class DiceFocalGSLLoss(nn.Module):
    """Loss schedulata Dice-Focal + GSL (Eq. 8), N-dimensionale.

    Identica in logica al vecchio progetto: alpha(t) parte da 1 (solo region
    loss) e scende verso 0 (GSL prevalente) secondo lo scheduler.

    Args:
        num_classes:            Numero di classi.
        gsl_class_weights:       Pesi globali w_k per la GSL.
        scheduler:               AlphaScheduler (default: step, h=5, 30 epoche).
        dice_weight/focal_weight/gamma/ignore_background: passati a CombinedLoss.
        region_class_weights:    Pesi per-classe della region loss (puo'
                                 differire dai pesi GSL).
        gsl_include_background:  Se la GSL somma anche sulla classe 0.
    """

    def __init__(
        self,
        num_classes: int,
        gsl_class_weights: Optional[Sequence[float]] = None,
        scheduler: Optional[AlphaScheduler] = None,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        ignore_background: bool = True,
        region_class_weights: Optional[Sequence[float]] = None,
        gsl_include_background: bool = True,
    ) -> None:
        super().__init__()
        self.region_loss = CombinedLoss(
            num_classes=num_classes,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            gamma=gamma,
            ignore_background=ignore_background,
            class_weights=region_class_weights,
        )
        self.gsl = GeneralizedSurfaceLoss(
            num_classes=num_classes,
            class_weights=gsl_class_weights,
            include_background=gsl_include_background,
        )
        self.scheduler = scheduler if scheduler is not None else AlphaScheduler(
            schedule="step", total_epochs=30, step_length=5
        )
        self._alpha: float = self.scheduler(0)

    def set_epoch(self, epoch: int) -> float:
        """Aggiorna alpha per l'epoca corrente (0-based). Ritorna il nuovo valore."""
        self._alpha = self.scheduler(epoch)
        return self._alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, dtm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcola la loss combinata schedulata.

        Returns:
            total, region, gsl, alpha_t (tutti scalari; gli ultimi 3 per logging).
        """
        region, _d, _f = self.region_loss(logits, targets)
        gsl = self.gsl(logits, targets, dtm)
        a = self._alpha
        total = a * region + (1.0 - a) * gsl
        alpha_t = torch.as_tensor(a, dtype=total.dtype, device=total.device)
        return total, region, gsl, alpha_t
