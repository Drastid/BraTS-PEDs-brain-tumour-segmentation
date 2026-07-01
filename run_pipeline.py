#!/usr/bin/env python
"""
run_pipeline.py
===============
Orchestratore end-to-end della pipeline BraTS-PEDs — un solo comando per:

    1. (opzionale) applicare la configurazione di scaling A100 (`documenti md/modifiche.md`)
    2. allenare i modelli baseline (U-Net, FPN, SegFormer) con loss CombinedLoss
    3. (opzionale) ramo Fase 3 — loss schedulata Dice-Focal + GSL (`documenti md/fase3_loss_gsl.md`)
    4. valutazione 3D sul test set (Dice / IoU / HD95)
    5. (opzionale) copia dei checkpoint su una cartella di backup (es. Google Drive)

Pensato sia per Colab/A100 sia per esecuzione locale. Riusa **esattamente** le
utility del progetto (`src/`), quindi rispetta automaticamente gli invarianti di
`CLAUDE.md`: augmentation solo geometrica, center-crop condiviso train/eval,
split congelato in `split.json`, loss in fp32 sotto AMP, determinismo completo.

--------------------------------------------------------------------------------
ESEMPI
--------------------------------------------------------------------------------
# Tutto, con i parametri A100 (B), dataset su NVMe locale, backup su Drive:
python run_pipeline.py \
    --data-root /content/processed_dataset \
    --ckpt-root /content/checkpoints \
    --apply-a100-config \
    --models unet fpn segformer \
    --evaluate \
    --backup-dir /content/drive/MyDrive/BraTS_Colab/checkpoints_a100

# Smoke test rapido (config A leggera, 1 epoca totale, un solo modello):
python run_pipeline.py --smoke-test --models unet

# Solo valutazione 3D (checkpoint già presenti):
python run_pipeline.py --evaluate --no-train

# Ramo GSL (Fase 3) su U-Net, DTM già scompattate in <split>/dtms:
python run_pipeline.py --models unet --gsl --no-eval

--------------------------------------------------------------------------------
NOTE
--------------------------------------------------------------------------------
* `--apply-a100-config` EDITA i file sorgente (`src/config.py`, `src/constants.py`,
  `src/evaluate_3d_test.py`) per portarli ai valori A100 (batch 64, crop 224,
  resnet50, mit-b2, bf16, lr 6e-4/2e-4) e per allineare l'encoder/variante usati
  in valutazione. È idempotente. Senza il flag, la pipeline usa i valori già
  presenti in `config.py`/`constants.py`.
* Cambiando encoder/crop, i VECCHI checkpoint (resnet34/mit-b1/crop192) NON sono
  riutilizzabili: questo script li riallena.
* bf16 non usa GradScaler (Ampere ha range fp32): su GPU non-Ampere o CPU lo
  script ricade automaticamente su fp16/none.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Project root on sys.path (così `import src...` funziona ovunque)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================== #
# 1. Applicazione (opzionale) della config A100 — edita i sorgenti
# =========================================================================== #

# (pattern_regex, replacement) per src/config.py — valori "config B" di modifiche.md §9
_CONFIG_EDITS = [
    (r"batch_size: int = \d+",                              "batch_size: int = 64"),
    (r'encoder: str = "[^"]*"',                             'encoder: str = "resnet50"'),
    (r'segformer_checkpoint: str = "[^"]*"',               'segformer_checkpoint: str = "nvidia/mit-b2"'),
    (r"num_workers: int = \d+",                             "num_workers: int = 8"),
    (r"lr_phase1: float = [0-9.eE+-]+",                     "lr_phase1: float = 6e-4"),
    (r"lr_phase2: float = [0-9.eE+-]+",                     "lr_phase2: float = 2e-4"),
    (r'amp_dtype: str = "[^"]*"',                           'amp_dtype: str = "bf16"'),
]
_CONSTANTS_EDITS = [
    (r"CROP_SIZE: int = \d+",                               "CROP_SIZE: int = 224"),
]
# Allineamento valutazione 3D (modifiche.md §3, §9 punti 10-11) — DEVE combaciare col training
_EVAL_EDITS = [
    (r'ENCODER = "[^"]*"',                                  'ENCODER = "resnet50"'),
    (r'model_checkpoint="nvidia/mit-b\d"',                  'model_checkpoint="nvidia/mit-b2"'),
]


def _apply_edits(path: Path, edits: List[tuple]) -> None:
    """Applica una lista di (regex, replacement) a un file di testo, in-place."""
    text = path.read_text(encoding="utf-8")
    for pattern, repl in edits:
        text = re.sub(pattern, repl, text)
    path.write_text(text, encoding="utf-8")


def apply_a100_config() -> None:
    """Porta src/config.py, src/constants.py, src/evaluate_3d_test.py ai valori A100.

    Idempotente: rieseguirlo non cambia nulla se i valori sono già quelli A100.
    """
    print("=" * 70)
    print("  [config] Applico la configurazione di scaling A100 (modifiche.md)")
    print("=" * 70)
    _apply_edits(PROJECT_ROOT / "src" / "config.py", _CONFIG_EDITS)
    _apply_edits(PROJECT_ROOT / "src" / "constants.py", _CONSTANTS_EDITS)
    _apply_edits(PROJECT_ROOT / "src" / "evaluate_3d_test.py", _EVAL_EDITS)
    print("  config.py        → batch=64, encoder=resnet50, segformer=mit-b2,")
    print("                     num_workers=8, lr=6e-4/2e-4, amp=bf16")
    print("  constants.py     → CROP_SIZE=224 (divisibile per 32, SegFormer-safe)")
    print("  evaluate_3d_test → ENCODER=resnet50, segformer=mit-b2 (allineati al training)")
    print()


# =========================================================================== #
# 2. Costruzione modello / ottimizzatore / scheduler
# =========================================================================== #

def build_model(arch: str, cfg, device):
    """Istanzia uno dei tre modelli, drop-in compatibili con il training loop."""
    import segmentation_models_pytorch as smp

    if arch == "unet":
        return smp.Unet(encoder_name=cfg.encoder, encoder_weights=cfg.encoder_weights,
                        in_channels=cfg.in_channels, classes=cfg.num_classes,
                        activation=None).to(device)
    if arch == "fpn":
        return smp.FPN(encoder_name=cfg.encoder, encoder_weights=cfg.encoder_weights,
                       in_channels=cfg.in_channels, classes=cfg.num_classes,
                       activation=None).to(device)
    if arch == "segformer":
        _check_transformers_version()
        from src.models import get_segformer
        return get_segformer(model_checkpoint=cfg.segformer_checkpoint,
                             num_classes=cfg.num_classes).to(device)
    raise ValueError(f"arch sconosciuta: {arch!r} (attese: unet|fpn|segformer)")


# Serie di transformers supportata da src/models.py. Il wrapper SegFormer
# gestisce SIA la struttura vecchia (segformer.encoder.patch_embeddings[i], <=~5.6)
# SIA quella nuova (segformer.stages[i].patch_embeddings, 5.7+), quindi qualunque
# 5.x va bene (5.6.2 in locale, 5.12.1 su Colab). Una 6.x potrebbe ristrutturare
# di nuovo → fuori dalla serie 5 avvisiamo. Vedi requirements.txt (transformers>=5,<6).
_SUPPORTED_TRANSFORMERS_MAJOR = 5


def _check_transformers_version() -> None:
    """Avvisa (non blocca) se transformers è fuori dalla serie 5.x supportata."""
    try:
        import transformers
    except ImportError:
        raise ImportError(
            'transformers non installato. Esegui: pip install "transformers>=5,<6"'
        )
    ver = transformers.__version__
    try:
        major = int(ver.split(".")[0])
    except (ValueError, IndexError):
        major = -1
    if major != _SUPPORTED_TRANSFORMERS_MAJOR:
        print(
            "\n" + "!" * 70 + "\n"
            f"  [ATTENZIONE] transformers {ver}: fuori dalla serie "
            f"{_SUPPORTED_TRANSFORMERS_MAJOR}.x supportata da src/models.py.\n"
            "  SegFormer potrebbe rompersi (struttura interna diversa). Consigliato:\n"
            '    pip install "transformers>=5,<6"  e RIAVVIA il runtime.\n'
            + "!" * 70 + "\n"
        )


def _monitor_value(history, window: int) -> float:
    """Moving-average of the last ``window`` val fg-Dice values in ``history``.

    Smooths the noisy BraTS-PEDs validation curve before it drives the
    early-stop decision and the best-checkpoint selection. ``window=1`` returns
    the raw latest value (legacy behaviour, exact reproducibility). The average
    is taken over whatever epochs exist so far (so early epochs use fewer
    points), spanning the Phase 1->2 boundary harmlessly since it is just the
    validation curve.

    Args:
        history: List of per-epoch rows (each with a ``val_fg_dice`` key).
        window:  Number of trailing epochs to average (>=1).

    Returns:
        The smoothed monitor value for the most recent epoch.
    """
    w = max(1, int(window))
    vals = [r["val_fg_dice"] for r in history[-w:]]
    return sum(vals) / len(vals)


class _EarlyStopper:
    """Tracks the best monitored metric and decides when to stop.

    Designed for a MAXIMISED metric (validation foreground Dice). ``update``
    is called once per epoch with the current value and returns ``True`` when
    training should stop, i.e. the metric has failed to improve by more than
    ``min_delta`` for ``patience`` consecutive *eligible* epochs.

    Eligibility is controlled by the caller: we only feed Phase-2 epochs to the
    stopper (Phase 1 is a fixed warm-up), while the best-checkpoint selection
    stays global across both phases in the training loop. ``patience <= 0`` or
    ``enabled=False`` disables stopping entirely (``update`` always returns
    False), so the flag is a no-op unless explicitly turned on.

    Args:
        enabled:   Master switch; when False the stopper never fires.
        patience:  Consecutive non-improving epochs tolerated before stopping.
        min_delta: Minimum increase over the running best to count as progress.
    """

    def __init__(self, enabled: bool, patience: int, min_delta: float = 0.0) -> None:
        self.enabled = bool(enabled) and patience > 0
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float = float("-inf")
        self.num_bad: int = 0
        self.best_epoch: int = -1

    def update(self, value: float, epoch: int) -> bool:
        """Register an epoch's metric; return True if training should stop."""
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
        if not self.enabled:
            return False
        return self.num_bad >= self.patience


def build_optim_sched(model, cfg, phase: int):
    """Two-phase schedule: Phase 1 encoder frozen, Phase 2 LR differenziale.

    Ricalca la metodologia dei notebook (CLAUDE.md §5): AdamW + CosineAnnealingLR
    per fase, encoder LR = lr_phase2 * encoder_lr_mult.
    """
    import torch
    from src.train_utils import set_encoder_trainable

    if phase == 1:
        set_encoder_trainable(model, False)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_phase1,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase1_epochs)
    else:
        set_encoder_trainable(model, True)
        enc = list(model.encoder.parameters())
        # Escludi i parametri encoder per IDENTITA' (non per prefisso nome): il
        # wrapper SegFormer espone i param sotto "model.*", quindi un filtro per
        # nome duplicherebbe il backbone tra i due gruppi e AdamW andrebbe in
        # errore. Per id funziona anche per i modelli SMP (Unet/FPN) invariati.
        enc_ids = {id(p) for p in enc}
        dec = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW(
            [{"params": enc, "lr": cfg.lr_phase2 * cfg.encoder_lr_mult},
             {"params": dec, "lr": cfg.lr_phase2}],
            weight_decay=cfg.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase2_epochs)
    return opt, sched


# =========================================================================== #
# 3. DataLoader
# =========================================================================== #

def make_loaders(cfg, data_root: str, return_dtm: bool, dtm_subdir: str = "dtms"):
    """Costruisce train/val DataLoader deterministici, con efficienza A100.

    Aggiunge persistent_workers/prefetch_factor (modifiche.md §6) solo quando
    num_workers > 0. Con return_dtm=True usa l'augmentation rigida DTM-safe e
    carica le DTM da <split>/<dtm_subdir>.
    """
    from torch.utils.data import DataLoader
    from src.dataset import BraTSDataset
    from src.train_utils import get_augmentation, make_generator, seed_worker

    aug = get_augmentation(
        p=cfg.augment_prob,
        with_dtm=return_dtm,
        strength=getattr(cfg, "augment_strength", 1.0),
        intensity=getattr(cfg, "augment_intensity", False),
    )

    def _ds(split, augment):
        kwargs = {}
        if return_dtm:
            kwargs = dict(return_dtm=True,
                          dtm_dir=os.path.join(data_root, split, dtm_subdir))
        return BraTSDataset(os.path.join(data_root, split), augment=augment, **kwargs)

    train_ds = _ds("train", aug)
    val_ds = _ds("val", None)

    extra = {}
    if cfg.num_workers > 0:
        extra = dict(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        worker_init_fn=seed_worker, generator=make_generator(cfg.seed), **extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, **extra,
    )
    return train_loader, val_loader


# =========================================================================== #
# 4. Training di un singolo modello (baseline o GSL)
# =========================================================================== #

def _resolve_amp(cfg, device) -> str:
    """bf16 solo su Ampere+; altrimenti fp16 su CUDA, none su CPU."""
    import torch
    want = cfg.amp_dtype
    if device.type != "cuda":
        return "none"
    if want == "bf16" and not torch.cuda.is_bf16_supported():
        print("  [amp] bf16 non supportato dalla GPU → ripiego su fp16.")
        return "fp16"
    return want


def train_model(arch: str, cfg, device, data_root: str, ckpt_root: str,
                use_gsl: bool, backup_dir: Optional[str],
                gsl_alpha_min: float = 0.2,
                run_name: Optional[str] = None) -> Dict[str, float]:
    """Allena un modello e salva best/last. Ritorna le metriche di validazione best."""
    import torch
    from src.train_utils import (set_seed, get_class_weights, save_checkpoint,
                                  train_one_epoch, evaluate, format_metrics)

    set_seed(cfg.seed, deterministic=True)
    amp_dtype = _resolve_amp(cfg, device)
    # GradScaler solo per fp16 reale
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == "fp16")) if amp_dtype == "fp16" else None

    print("\n" + "#" * 70)
    print(f"  TRAIN  {arch.upper()}  |  gsl={use_gsl}  amp={amp_dtype}  "
          f"batch={cfg.batch_size}  crop={cfg.crop_size}  epoche={cfg.total_epochs}")
    print("#" * 70)

    model = build_model(arch, cfg, device)
    ckpt_dir = os.path.join(ckpt_root, arch)
    os.makedirs(ckpt_dir, exist_ok=True)

    if use_gsl:
        return _train_gsl(arch, model, cfg, device, data_root, ckpt_dir,
                          amp_dtype, scaler, backup_dir, alpha_min=gsl_alpha_min,
                          run_name=run_name)

    # --- Baseline: CombinedLoss (gamma=, class_weights=) ---
    from src.losses import CombinedLoss
    criterion = CombinedLoss(
        num_classes=cfg.num_classes,
        dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight,
        gamma=cfg.focal_gamma, ignore_background=cfg.ignore_background,
        class_weights=_train_class_weights(data_root, device, cfg.num_classes),
    ).to(device)

    train_loader, val_loader = make_loaders(cfg, data_root, return_dtm=False)

    W = max(1, getattr(cfg, "es_smooth_window", 1))
    stopper = _EarlyStopper(cfg.early_stopping, cfg.es_patience, cfg.es_min_delta)
    if cfg.early_stopping:
        _mon = f"val_fg_dice (media mobile {W})" if W > 1 else "val_fg_dice"
        print(f"  [early-stopping] ON — monitor={_mon} patience={cfg.es_patience} "
              f"min_delta={cfg.es_min_delta} (Phase 2 only)")
    elif W > 1:
        print(f"  [monitor] best.pth selezionato su val_fg_dice, media mobile a {W} epoche")

    best_fg, best_mon, best_metrics, g, history = -1.0, -1.0, {}, 0, []
    stopped = False
    for phase, n_ep in [(1, cfg.phase1_epochs), (2, cfg.phase2_epochs)]:
        opt, sched = build_optim_sched(model, cfg, phase)
        for e in range(n_ep):
            tr = train_one_epoch(model, train_loader, criterion, opt, device,
                                 crop_size=cfg.crop_size, scaler=scaler, amp_dtype=amp_dtype)
            va = evaluate(model, val_loader, criterion, device, crop_size=cfg.crop_size)
            sched.step()
            fg = (va["dice_NCR"] + va["dice_ED"] + va["dice_ET"]) / 3.0
            history.append(_history_row(g, phase, tr, va, fg))
            mon = _monitor_value(history, W)
            print(f"  [P{phase} ep {e+1}/{n_ep}] {format_metrics(va, 'val')}  fg_dice={fg:.4f}")
            if mon > best_mon:
                best_mon, best_fg, best_metrics = mon, fg, va
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, opt, sched, g, va)
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, opt, sched, g, va)
            g += 1
            # Early stopping is evaluated only during Phase 2 (Phase 1 is a
            # fixed warm-up); the global best above is untouched by it.
            if phase == 2 and stopper.update(mon, g - 1):
                print(f"  [early-stopping] stop a epoch {g-1}: nessun miglioramento "
                      f"del val fg-Dice per {cfg.es_patience} epoche "
                      f"(best={stopper.best:.4f} @ ep{stopper.best_epoch}).")
                stopped = True
                break
        if stopped:
            break
    _save_history(ckpt_dir, history)

    _report_vram(device)
    print(f"  → best FG Dice ({arch}): {best_fg:.4f}")
    if backup_dir:
        _backup(ckpt_dir, backup_dir, arch, run_name)
    return best_metrics


class _Phase2AlphaScheduler:
    """AlphaScheduler che resta a 1.0 in Phase 1 e scende (con floor) in Phase 2.

    Due scelte di design (decise con l'utente), realizzate SENZA toccare
    ``src/losses.py``:

    1. **alpha=1.0 fisso durante Phase 1** (encoder congelato): la GSL inizia a
       contare solo quando l'encoder può davvero adattarsi ai bordi. Lo
       scheduler interno è costruito sulle SOLE epoche di Phase 2; per gli indici
       di epoca < ``phase1_epochs`` ritorna 1.0.

    2. **floor su alpha** (default 0.2): in Phase 2 alpha scende da 1.0 ma non
       sotto ``alpha_min``, così la region loss (Dice+Focal) resta sempre viva
       (es. 20%) ed evita il collasso del gradiente quando la GSL è già minima.
       Deviazione consapevole dal paper (che porta alpha a 0) — da dichiarare nel
       report.

    L'oggetto è compatibile con ``DiceFocalGSLLoss`` (espone ``__call__(epoch)``);
    ``set_epoch`` della loss lo userà come un normale AlphaScheduler.
    """

    def __init__(self, inner, phase1_epochs: int, alpha_min: float = 0.2) -> None:
        self.inner = inner                       # AlphaScheduler sulle epoche di Phase 2
        self.phase1_epochs = int(phase1_epochs)
        self.alpha_min = float(alpha_min)

    def __call__(self, epoch: int) -> float:
        if epoch < self.phase1_epochs:
            return 1.0                            # Phase 1: solo region loss
        # Rebase: l'epoca 0 di Phase 2 mappa sull'epoca 0 dello scheduler interno
        a = self.inner(epoch - self.phase1_epochs)
        return max(self.alpha_min, a)             # floor: region sempre >= alpha_min


def _train_gsl(arch, model, cfg, device, data_root, ckpt_dir,
               amp_dtype, scaler, backup_dir, alpha_min: float = 0.2,
               run_name: Optional[str] = None) -> Dict[str, float]:
    """Training con loss schedulata Dice-Focal + GSL (fase3_loss_gsl.md §3).

    alpha=1.0 in Phase 1 (encoder frozen), poi scende a gradini su Phase 2 con un
    floor ``alpha_min`` (vedi :class:`_Phase2AlphaScheduler`).
    """
    import json
    from src.losses import DiceFocalGSLLoss, AlphaScheduler
    from src.train_utils import train_one_epoch_gsl, evaluate_gsl, format_metrics, save_checkpoint

    weights_path = os.path.join(data_root, "gsl_class_weights.json")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Pesi GSL non trovati: {weights_path}\n"
            "Esegui prima il pre-calcolo: `python -m scripts.precompute_gsl_stats --archive tar`\n"
            "(o lancia run_pipeline.py con --gsl-precompute)."
        )
    with open(weights_path) as f:
        wk = json.load(f)["weights"]

    # Scheduler interno sulle SOLE epoche di Phase 2; il wrapper aggiunge il
    # plateau di Phase 1 (alpha=1) e il floor.
    inner = AlphaScheduler(schedule="step", total_epochs=cfg.phase2_epochs, step_length=5)
    scheduler = _Phase2AlphaScheduler(inner, phase1_epochs=cfg.phase1_epochs, alpha_min=alpha_min)

    criterion = DiceFocalGSLLoss(
        num_classes=cfg.num_classes, gsl_class_weights=wk,
        scheduler=scheduler,
        dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight,
        gamma=cfg.focal_gamma, ignore_background=cfg.ignore_background,
    ).to(device)
    print(f"  [gsl] alpha: 1.0 in Phase 1 ({cfg.phase1_epochs} ep), poi step-5 "
          f"in Phase 2 ({cfg.phase2_epochs} ep) con floor={alpha_min}")

    train_loader, val_loader = make_loaders(cfg, data_root, return_dtm=True)

    W = max(1, getattr(cfg, "es_smooth_window", 1))
    stopper = _EarlyStopper(cfg.early_stopping, cfg.es_patience, cfg.es_min_delta)
    if cfg.early_stopping:
        _mon = f"val_fg_dice (media mobile {W})" if W > 1 else "val_fg_dice"
        print(f"  [early-stopping] ON — monitor={_mon} patience={cfg.es_patience} "
              f"min_delta={cfg.es_min_delta} (Phase 2 only)")
    elif W > 1:
        print(f"  [monitor] best.pth selezionato su val_fg_dice, media mobile a {W} epoche")

    best_fg, best_mon, best_metrics, g, history = -1.0, -1.0, {}, 0, []
    stopped = False
    for phase, n_ep in [(1, cfg.phase1_epochs), (2, cfg.phase2_epochs)]:
        opt, sched = build_optim_sched(model, cfg, phase)
        for e in range(n_ep):
            tr = train_one_epoch_gsl(model, train_loader, criterion, opt, device,
                                     epoch=g, crop_size=cfg.crop_size,
                                     scaler=scaler, amp_dtype=amp_dtype)
            va = evaluate_gsl(model, val_loader, criterion, device, crop_size=cfg.crop_size)
            sched.step()
            fg = (va["dice_NCR"] + va["dice_ED"] + va["dice_ET"]) / 3.0
            history.append(_history_row(g, phase, tr, va, fg))
            mon = _monitor_value(history, W)
            print(f"  [P{phase} ep {e+1}/{n_ep}] alpha={va.get('alpha', float('nan')):.2f}  "
                  f"{format_metrics(va, 'val')}  fg_dice={fg:.4f}")
            if mon > best_mon:
                best_mon, best_fg, best_metrics = mon, fg, va
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, opt, sched, g, va)
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, opt, sched, g, va)
            g += 1
            if phase == 2 and stopper.update(mon, g - 1):
                print(f"  [early-stopping] stop a epoch {g-1}: nessun miglioramento "
                      f"del val fg-Dice per {cfg.es_patience} epoche "
                      f"(best={stopper.best:.4f} @ ep{stopper.best_epoch}).")
                stopped = True
                break
        if stopped:
            break
    _save_history(ckpt_dir, history)

    _report_vram(device)
    print(f"  → best FG Dice GSL ({arch}): {best_fg:.4f}")
    if backup_dir:
        _backup(ckpt_dir, backup_dir, arch, run_name)
    return best_metrics


# =========================================================================== #
# 5. Helpers: VRAM, backup, GSL precompute, eval
# =========================================================================== #

def _report_vram(device) -> None:
    import torch
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [vram] picco allocato: {peak:.1f} GB")
        torch.cuda.reset_peak_memory_stats()


def _backup(ckpt_dir: str, backup_dir: str, arch: str,
            run_name: Optional[str] = None) -> None:
    """Copia i checkpoint di un modello nella cartella di backup (es. Drive).

    Quando ``run_name`` e' impostato, la destinazione e' namespacizzata come
    ``backup_dir/<run_name>/<arch>``: senza questo, run diversi scriverebbero
    tutti in ``backup_dir/<arch>`` e la ``history.json`` (e i .pth) dell'ultimo
    sovrascriverebbe quelli dei precedenti, rendendo i run non confrontabili.
    """
    dst = os.path.join(backup_dir, run_name, arch) if run_name \
        else os.path.join(backup_dir, arch)
    os.makedirs(dst, exist_ok=True)
    for name in ("best.pth", "last.pth", "history.json"):
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    print(f"  [backup] {ckpt_dir} → {dst}")


def _train_class_weights(data_root: str, device, num_classes: int):
    """Pesi inverse-frequency calcolati SOLO sul TRAIN split (no leak val/test).

    Conta i voxel per classe sulle maschere di ``<data_root>/train/masks``, li
    converte in frequenze e poi in pesi via ``get_class_weights``. Risultato in
    cache su ``<data_root>/class_freq_train.json``. Se le maschere non ci sono,
    ripiega su ``constants.VOXEL_FREQ`` (statistica globale) documentando il caso.
    """
    import glob
    import json
    import numpy as np
    from src.train_utils import get_class_weights

    cache = os.path.join(data_root, "class_freq_train.json")
    if os.path.isfile(cache):
        freq = np.asarray(json.load(open(cache))["freq"], dtype=np.float32)
        print(f"  [class-weights] frequenze TRAIN-only da cache: {np.round(freq, 5)}")
        return get_class_weights(voxel_freq=freq, device=device)

    mask_dir = os.path.join(data_root, "train", "masks")
    files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
    if not files:
        print(f"  [class-weights] {mask_dir} assente/vuota -> fallback su VOXEL_FREQ (constants).")
        return get_class_weights(device=device)

    counts = np.zeros(num_classes, dtype=np.int64)
    for fpath in files:
        m = np.load(fpath).astype(np.int64).ravel()
        counts += np.bincount(m, minlength=num_classes)[:num_classes]
    freq = (counts / counts.sum()).astype(np.float32)
    with open(cache, "w") as f:
        json.dump({"split": "train", "n_masks": len(files),
                   "counts": counts.tolist(), "freq": freq.tolist()}, f, indent=2)
    print(f"  [class-weights] TRAIN-only ({len(files)} maschere) freq={np.round(freq, 5)} "
          f"-> cache {cache}")
    return get_class_weights(voxel_freq=freq, device=device)


def _history_row(g: int, phase: int, tr: Dict[str, float],
                 va: Dict[str, float], val_fg: float) -> Dict[str, float]:
    """Una riga di history per epoca: loss e Dice foreground, train E val."""
    def _fg(d):
        return (d.get("dice_NCR", 0.0) + d.get("dice_ED", 0.0) + d.get("dice_ET", 0.0)) / 3.0
    row = {"epoch": g, "phase": phase,
           "train_loss": tr.get("loss"), "val_loss": va.get("loss"),
           "train_fg_dice": _fg(tr), "val_fg_dice": val_fg}
    for k in ("dice_NCR", "dice_ED", "dice_ET", "dice_background"):
        row[f"train_{k}"] = tr.get(k)
        row[f"val_{k}"] = va.get(k)
    if "alpha" in va:
        row["alpha"] = va.get("alpha")
    return row


def _save_history(ckpt_dir: str, history: List[dict]) -> None:
    """Scrive le curve train/val per epoca in <ckpt_dir>/history.json."""
    import json
    path = os.path.join(ckpt_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [history] curve train/val -> {path} ({len(history)} epoche)")


def gsl_precompute(data_root: str, archive: str) -> None:
    """Lancia lo script offline GSL (pesi globali + DTM per slice)."""
    print("\n" + "=" * 70)
    print("  [gsl] Pre-calcolo pesi globali + DTM (precompute_gsl_stats)")
    print("=" * 70)
    cmd = [sys.executable, "-m", "scripts.precompute_gsl_stats",
           "--data-root", data_root]
    if archive != "none":
        cmd += ["--archive", archive]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def run_evaluation(models: List[str], data_root: str, ckpt_root: str,
                   output_dir: Optional[str] = None) -> None:
    """Valutazione 3D sul test set via lo script CLI del progetto.

    Inoltra --data-root/--ckpt-root (e opzionale --output-dir) a
    src.evaluate_3d_test, così l'eval usa ESATTAMENTE gli stessi path del
    training (niente più path hardcoded: baseline e GSL restano separati).
    """
    print("\n" + "=" * 70)
    print("  [eval] Valutazione 3D sul TEST set (Dice / IoU / HD95)")
    print("=" * 70)
    # 'all' se ci sono tutti e tre, altrimenti un modello alla volta
    targets = ["all"] if set(models) >= {"unet", "fpn", "segformer"} else models
    for tg in targets:
        cmd = [sys.executable, "-m", "src.evaluate_3d_test", "--model", tg,
               "--data-root", data_root, "--ckpt-root", ckpt_root]
        if output_dir:
            cmd += ["--output-dir", output_dir]
        print("  $ " + " ".join(cmd))
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


# =========================================================================== #
# 6. Main
# =========================================================================== #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestratore end-to-end della pipeline BraTS-PEDs (Colab/A100-ready).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "processed_dataset"),
                   help="Root del dataset pre-processato (su Colab: /content/processed_dataset).")
    p.add_argument("--ckpt-root", default=str(PROJECT_ROOT / "checkpoints"),
                   help="Dove salvare i checkpoint (su Colab: /content/checkpoints).")
    p.add_argument("--models", nargs="+", default=["unet", "fpn", "segformer"],
                   choices=["unet", "fpn", "segformer"],
                   help="Modelli da allenare/valutare.")

    p.add_argument("--apply-a100-config", action="store_true",
                   help="Edita src/config.py, constants.py, evaluate_3d_test.py ai valori A100.")

    # Train on/off
    p.add_argument("--no-train", dest="train", action="store_false",
                   help="Salta il training (utile per sola valutazione).")
    p.set_defaults(train=True)

    # Eval on/off (default: on)
    p.add_argument("--evaluate", dest="evaluate", action="store_true",
                   help="Esegui la valutazione 3D dopo il training (default).")
    p.add_argument("--no-eval", dest="evaluate", action="store_false",
                   help="Salta la valutazione 3D.")
    p.set_defaults(evaluate=True)

    # GSL (Fase 3)
    p.add_argument("--gsl", action="store_true",
                   help="Usa la loss schedulata Dice-Focal + GSL invece di CombinedLoss.")
    p.add_argument("--gsl-precompute", action="store_true",
                   help="Prima del training GSL, (ri)genera pesi globali + DTM.")
    p.add_argument("--gsl-archive", default="tar", choices=["none", "tar", "zip"],
                   help="Formato archivio DTM passato a precompute_gsl_stats.")
    p.add_argument("--gsl-alpha-min", type=float, default=0.2,
                   help="Floor di alpha in Phase 2 (la region Dice+Focal resta viva a "
                        "questa frazione). 0.0 = fedele al paper (GSL pura a fine training). "
                        "alpha resta 1.0 durante Phase 1 (encoder frozen).")

    p.add_argument("--backup-dir", default=None,
                   help="Cartella (es. su Drive) dove copiare i checkpoint dopo ogni modello.")

    # Override veloci
    p.add_argument("--epochs", type=int, default=None,
                   help="Override del numero di epoche di Phase 2 (default: dalla config).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override del batch size (default: dalla config).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run minimo: 1 epoca per fase, batch ridotto, encoder leggero (config A). "
                        "Per verificare che tutto giri prima del run vero.")

    # ── Run isolation (per confronto tra prove, senza sovrascrivere) ──────
    p.add_argument("--run-name", default=None,
                   help="Etichetta della prova. Se impostata, i checkpoint vanno in "
                        "<ckpt-root>/<run-name>/<arch> e le metriche 3D in "
                        "evaluation_outputs/<run-name>/, cosi' i run vecchi NON vengono "
                        "sovrascritti e restano confrontabili.")

    # ── Leve anti-overfitting (parametriche) ─────────────────────────────
    p.add_argument("--early-stopping", dest="early_stopping", action="store_true",
                   help="Attiva early stopping sul val fg-Dice (solo Phase 2).")
    p.add_argument("--es-patience", type=int, default=None,
                   help="Epoche senza miglioramento prima di fermarsi (default: config=6).")
    p.add_argument("--es-min-delta", type=float, default=None,
                   help="Miglioramento minimo del val fg-Dice per contare come progresso.")
    p.add_argument("--es-smooth-window", type=int, default=None,
                   help="Finestra media-mobile sul val fg-Dice per early stop E scelta "
                        "del best.pth (1=nessuno smoothing; 3 consigliato per la val rumorosa).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override del seed (per ripetizioni multi-seed).")

    p.add_argument("--weight-decay", type=float, default=None,
                   help="Override del weight decay AdamW (regolarizzazione L2).")
    p.add_argument("--encoder-lr-mult", type=float, default=None,
                   help="Moltiplicatore LR encoder in Phase 2 (piu' basso = encoder piu' frenato).")
    p.add_argument("--lr-phase2", type=float, default=None,
                   help="Override dell'LR base di Phase 2.")
    p.add_argument("--phase1-epochs", type=int, default=None,
                   help="Override delle epoche di Phase 1 (warm-up).")
    p.add_argument("--phase2-epochs", type=int, default=None,
                   help="Override delle epoche di Phase 2 (alias di --epochs).")

    p.add_argument("--augment-prob", type=float, default=None,
                   help="Probabilita' per-transform dell'augmentation (default: config=0.5).")
    p.add_argument("--augment-strength", type=float, default=None,
                   help="Scala la magnitudine dell'augmentation geometrica (1.0=legacy, >1 piu' forte).")
    p.add_argument("--augment-intensity", dest="augment_intensity", action="store_true",
                   help="Aggiunge rumore/contrasto z-score-safe all'augmentation (solo baseline, no-DTM).")

    p.set_defaults(early_stopping=False, augment_intensity=False)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    # 1. Config A100 (prima di importare src.config, così i nuovi valori sono letti)
    if args.apply_a100_config:
        apply_a100_config()

    # Import differito: dopo l'eventuale edit dei sorgenti
    import torch
    from src.config import TrainConfig

    # Flag Ampere TF32 (modifiche.md §9 punto 12) — innocui altrove
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device={device} | "
          f"GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else '—'}")

    # 2. Costruisci la config (con override)
    overrides: Dict = {}
    if args.smoke_test:
        # Config A leggera + 1 epoca per fase: solo per verificare la pipeline.
        overrides.update(dict(encoder="resnet34", segformer_checkpoint="nvidia/mit-b1",
                              batch_size=8, num_workers=2,
                              phase1_epochs=1, phase2_epochs=1))
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    # --epochs e --phase2-epochs sono alias; --phase2-epochs ha la precedenza.
    if args.epochs is not None:
        overrides["phase2_epochs"] = args.epochs
    if args.phase2_epochs is not None:
        overrides["phase2_epochs"] = args.phase2_epochs
    if args.phase1_epochs is not None:
        overrides["phase1_epochs"] = args.phase1_epochs

    # Leve anti-overfitting
    if args.early_stopping:
        overrides["early_stopping"] = True
    if args.es_patience is not None:
        overrides["es_patience"] = args.es_patience
    if args.es_min_delta is not None:
        overrides["es_min_delta"] = args.es_min_delta
    if args.es_smooth_window is not None:
        overrides["es_smooth_window"] = args.es_smooth_window
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.encoder_lr_mult is not None:
        overrides["encoder_lr_mult"] = args.encoder_lr_mult
    if args.lr_phase2 is not None:
        overrides["lr_phase2"] = args.lr_phase2
    if args.augment_prob is not None:
        overrides["augment_prob"] = args.augment_prob
    if args.augment_strength is not None:
        overrides["augment_strength"] = args.augment_strength
    if args.augment_intensity:
        overrides["augment_intensity"] = True

    cfg = TrainConfig(**overrides)

    print(f"Config: batch={cfg.batch_size} crop={cfg.crop_size} enc={cfg.encoder} "
          f"seg={cfg.segformer_checkpoint} workers={cfg.num_workers} "
          f"lr={cfg.lr_phase1}/{cfg.lr_phase2} amp={cfg.amp_dtype} epoche={cfg.total_epochs}")

    if not os.path.isdir(args.data_root):
        print(f"\n[ERRORE] data-root inesistente: {args.data_root}\n"
              "Su Colab scompatta processed_dataset.zip in /content/ e passa "
              "--data-root /content/processed_dataset.")
        return 1

    # 3. (opzionale) pre-calcolo GSL
    if args.gsl and args.gsl_precompute:
        gsl_precompute(args.data_root, args.gsl_archive)

    # 3b. Isolamento del run: se --run-name e' impostato, i checkpoint e le
    #     metriche 3D vanno in sottocartelle dedicate, cosi' i run precedenti
    #     restano intatti e confrontabili.
    ckpt_root = args.ckpt_root
    eval_out_dir: Optional[str] = None
    if args.run_name:
        ckpt_root = os.path.join(args.ckpt_root, args.run_name)
        eval_out_dir = str(PROJECT_ROOT / "evaluation_outputs" / args.run_name)
        os.makedirs(ckpt_root, exist_ok=True)
        os.makedirs(eval_out_dir, exist_ok=True)
        # Registra la config esatta della prova accanto ai checkpoint.
        import json as _json
        run_meta = {"run_name": args.run_name, "models": args.models,
                    "gsl": args.gsl, "config": cfg.to_dict()}
        with open(os.path.join(ckpt_root, "run_config.json"), "w") as _f:
            _json.dump(run_meta, _f, indent=2)
        print(f"[run] '{args.run_name}': checkpoint -> {ckpt_root}")
        print(f"[run] '{args.run_name}': metriche 3D -> {eval_out_dir}")
        print(f"[run] config salvata -> {os.path.join(ckpt_root, 'run_config.json')}")
    else:
        print("[run] nessun --run-name: uso i path di default (ATTENZIONE: puo' "
              "sovrascrivere i run precedenti).")

    # 4. Training
    if args.train:
        results = {}
        for arch in args.models:
            results[arch] = train_model(
                arch, cfg, device, args.data_root, ckpt_root,
                use_gsl=args.gsl, backup_dir=args.backup_dir,
                gsl_alpha_min=args.gsl_alpha_min,
                run_name=args.run_name,
            )
        print("\n" + "=" * 70)
        print("  RIEPILOGO best FG Dice (val)")
        print("=" * 70)
        for arch, m in results.items():
            fg = (m.get("dice_NCR", 0) + m.get("dice_ED", 0) + m.get("dice_ET", 0)) / 3.0
            print(f"  {arch:10s}  fg_dice={fg:.4f}")
    else:
        print("\n[skip] training disabilitato (--no-train).")

    # 5. Valutazione 3D
    if args.evaluate:
        run_evaluation(args.models, args.data_root, ckpt_root, output_dir=eval_out_dir)
        if args.backup_dir:
            out = Path(eval_out_dir) if eval_out_dir else (PROJECT_ROOT / "evaluation_outputs")
            if out.is_dir():
                sub = os.path.join("evaluation_outputs", args.run_name) if args.run_name \
                    else "evaluation_outputs"
                dst = os.path.join(args.backup_dir, sub)
                shutil.copytree(out, dst, dirs_exist_ok=True)
                print(f"  [backup] {out} → {dst}")
    else:
        print("\n[skip] valutazione disabilitata (--no-eval).")

    print(f"\n[DONE] Pipeline completata in {(time.time()-t_start)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
