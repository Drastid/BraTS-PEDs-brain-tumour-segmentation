"""
src/train_3d.py
==================
Training e validation loop per i modelli 3D BraTS-PEDs (roadmap §5.3, fase
Colab). Due funzioni di training (una per ciascun ramo --loss) che condividono
la stessa logica di validazione via sliding_window_inference.

Design
------
- train_one_epoch_3d / train_one_epoch_gsl_3d: un'epoca sulle patch (ROI fisso,
  es. 128^3), loss in fp32 sotto autocast bf16 (A100-friendly, come nel vecchio
  progetto 2D).
- evaluate_3d: validazione sul VOLUME INTERO via sliding_window_inference
  (nessun patch sampling, nessun crop) — la GSL non si valuta qui (e' un
  termine di training; le metriche di validazione sono sempre Dice/HD95 via
  src/metrics_3d.py, indipendentemente dal ramo di training usato).
- MetricTracker, save_checkpoint, load_checkpoint: riusati concettualmente dal
  vecchio progetto 2D (stessa interfaccia), reimplementati qui per non
  introdurre una dipendenza diretta da master/ (workspace isolato, come da
  regola di ingaggio).
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses_3d import DiceFocalGSLLoss
from .metrics_3d import SegmentationMetrics3D


# ---------------------------------------------------------------------------
# Riproducibilita' (stessa politica del vecchio progetto 2D)
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fissa i seed di random/numpy/torch (CPU+CUDA). Vedi master/src/train_utils.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metric tracker (running mean)
# ---------------------------------------------------------------------------


class MetricTracker:
    """Accumula medie mobili per un insieme arbitrario di metriche scalari."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def means(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}


# ---------------------------------------------------------------------------
# Training — ramo baseline (DiceFocalLoss MONAI o CombinedLoss N-D)
# ---------------------------------------------------------------------------


def train_one_epoch_3d(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: str = "bf16",
) -> Dict[str, float]:
    """Un'epoca di training sulle patch 3D (ramo --loss=dice_focal).

    Args:
        model:     Modello 3D in modalita' train.
        loader:    DataLoader di training (patch-based, MONAI).
        criterion: Loss callable(logits, targets) -> scalar OPPURE
                   callable(...) -> (total, d_loss, f_loss) (CombinedLoss N-D).
                   Gestito dinamicamente in base al tipo di ritorno.
        optimizer: Ottimizzatore (tipicamente build_optimizer_3d).
        device:    Device CUDA.
        amp_dtype: "bf16" (raccomandato su A100) o "none" per disabilitare AMP.

    Returns:
        Dict di metriche medie sull'epoca: almeno "loss".
    """
    model.train()
    tracker = MetricTracker()
    use_amp = amp_dtype != "none"
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    for batch in tqdm(loader, desc="  train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True).squeeze(1).long()  # [B,*roi]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
            logits = model(images)

        loss_out = criterion(logits.float(), masks)
        if isinstance(loss_out, tuple):
            total = loss_out[0]
            batch_metrics = {"loss": total.item(), "dice_loss": loss_out[1].item(),
                              "focal_loss": loss_out[2].item()}
        else:
            total = loss_out
            batch_metrics = {"loss": total.item()}

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        tracker.update(batch_metrics, n=images.size(0))

    return tracker.means()


def train_one_epoch_gsl_3d(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceFocalGSLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp_dtype: str = "bf16",
) -> Dict[str, float]:
    """Un'epoca di training con la loss schedulata Dice-Focal + GSL (ramo --loss=gsl).

    Il DataLoader deve produrre batch con chiave "distance_map" (patch-level
    DTM, calcolata da ComputeDistanceMapd — si veda src/dataset_3d.py con
    with_dtm=True).

    Args:
        model:     Modello 3D in modalita' train.
        loader:    DataLoader di training con with_dtm=True.
        criterion: DiceFocalGSLLoss.
        optimizer: Ottimizzatore.
        device:    Device CUDA.
        epoch:     Indice di epoca 0-based — aggiorna alpha(t) UNA VOLTA per
                   epoca (non per batch), come nel vecchio progetto 2D.
        amp_dtype: "bf16" o "none".

    Returns:
        Dict di metriche medie: "loss", "region_loss", "gsl_loss", "alpha".
    """
    model.train()
    tracker = MetricTracker()
    use_amp = amp_dtype != "none"
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    criterion.set_epoch(epoch)

    for batch in tqdm(loader, desc="  train[gsl]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True).squeeze(1).long()
        dtms = batch["distance_map"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
            logits = model(images)

        total, region, gsl, alpha_t = criterion(logits.float(), masks, dtms.float())

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        tracker.update(
            {
                "loss": total.item(),
                "region_loss": region.item(),
                "gsl_loss": gsl.item(),
                "alpha": float(alpha_t.item()),
            },
            n=images.size(0),
        )

    return tracker.means()


# ---------------------------------------------------------------------------
# Validazione — SEMPRE su volume intero via sliding-window (nessun patch
# sampling, indipendentemente dal ramo di training usato).
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_3d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi: tuple[int, int, int],
    num_classes: int,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> Dict[str, float]:
    """Valuta il modello sul volume INTERO di ogni soggetto via sliding window.

    Le metriche riportate sono SEMPRE Dice + HD95 per-classe (src/metrics_3d.py),
    indipendentemente dal ramo di training (--loss=dice_focal o --loss=gsl): la
    GSL e' un termine di training, non una metrica clinica di validazione.

    Args:
        model:         Modello 3D in modalita' eval.
        loader:        DataLoader di validazione/test (volumi interi, batch_size=1).
        device:        Device CUDA.
        roi:           Dimensione della finestra scorrevole (stessa ROI del
                       training, es. (128,128,128)).
        num_classes:   Numero di classi (5).
        sw_batch_size: Numero di finestre processate in parallelo dalla sliding
                       window (non il batch di SOGGETTI: quello resta 1).
        overlap:       Sovrapposizione tra finestre adiacenti (0.5 = 50%,
                       blending "gaussian" di default in MONAI).

    Returns:
        Dict con dice_<classe>/hd95_<classe> per ET/NET/CC/ED, più
        dice_mean_fg/hd95_mean_fg.
    """
    model.eval()
    metrics = SegmentationMetrics3D(num_classes=num_classes)

    for batch in tqdm(loader, desc="  val  ", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True).squeeze(1).long()

        logits = sliding_window_inference(
            inputs=images, roi_size=roi, sw_batch_size=sw_batch_size,
            predictor=model, overlap=overlap, mode="gaussian",
        )
        metrics.update(logits, masks)

    return metrics.aggregate_and_reset()


# ---------------------------------------------------------------------------
# Checkpoint (stessa interfaccia del vecchio progetto 2D)
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Salva lo stato di training su disco."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "metrics": metrics},
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Ripristina lo stato di training da un checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
