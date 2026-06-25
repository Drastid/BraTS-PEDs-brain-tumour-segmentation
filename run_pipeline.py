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
        from src.models import get_segformer
        return get_segformer(model_checkpoint=cfg.segformer_checkpoint,
                             num_classes=cfg.num_classes).to(device)
    raise ValueError(f"arch sconosciuta: {arch!r} (attese: unet|fpn|segformer)")


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

    aug = get_augmentation(p=cfg.augment_prob, with_dtm=return_dtm)

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
                use_gsl: bool, backup_dir: Optional[str]) -> Dict[str, float]:
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
                          amp_dtype, scaler, backup_dir)

    # --- Baseline: CombinedLoss (gamma=, class_weights=) ---
    from src.losses import CombinedLoss
    criterion = CombinedLoss(
        num_classes=cfg.num_classes,
        dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight,
        gamma=cfg.focal_gamma, ignore_background=cfg.ignore_background,
        class_weights=get_class_weights(device=device),
    ).to(device)

    train_loader, val_loader = make_loaders(cfg, data_root, return_dtm=False)

    best_fg, best_metrics, g = -1.0, {}, 0
    for phase, n_ep in [(1, cfg.phase1_epochs), (2, cfg.phase2_epochs)]:
        opt, sched = build_optim_sched(model, cfg, phase)
        for e in range(n_ep):
            train_one_epoch(model, train_loader, criterion, opt, device,
                            crop_size=cfg.crop_size, scaler=scaler, amp_dtype=amp_dtype)
            va = evaluate(model, val_loader, criterion, device, crop_size=cfg.crop_size)
            sched.step()
            fg = (va["dice_NCR"] + va["dice_ED"] + va["dice_ET"]) / 3.0
            print(f"  [P{phase} ep {e+1}/{n_ep}] {format_metrics(va, 'val')}  fg_dice={fg:.4f}")
            if fg > best_fg:
                best_fg, best_metrics = fg, va
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, opt, sched, g, va)
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, opt, sched, g, va)
            g += 1

    _report_vram(device)
    print(f"  → best FG Dice ({arch}): {best_fg:.4f}")
    if backup_dir:
        _backup(ckpt_dir, backup_dir, arch)
    return best_metrics


def _train_gsl(arch, model, cfg, device, data_root, ckpt_dir,
               amp_dtype, scaler, backup_dir) -> Dict[str, float]:
    """Training con loss schedulata Dice-Focal + GSL (fase3_loss_gsl.md §3)."""
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

    criterion = DiceFocalGSLLoss(
        num_classes=cfg.num_classes, gsl_class_weights=wk,
        scheduler=AlphaScheduler(schedule="step", total_epochs=cfg.total_epochs, step_length=5),
        dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight,
        gamma=cfg.focal_gamma, ignore_background=cfg.ignore_background,
    ).to(device)

    train_loader, val_loader = make_loaders(cfg, data_root, return_dtm=True)

    best_fg, best_metrics, g = -1.0, {}, 0
    for phase, n_ep in [(1, cfg.phase1_epochs), (2, cfg.phase2_epochs)]:
        opt, sched = build_optim_sched(model, cfg, phase)
        for e in range(n_ep):
            train_one_epoch_gsl(model, train_loader, criterion, opt, device,
                                epoch=g, crop_size=cfg.crop_size,
                                scaler=scaler, amp_dtype=amp_dtype)
            va = evaluate_gsl(model, val_loader, criterion, device, crop_size=cfg.crop_size)
            sched.step()
            fg = (va["dice_NCR"] + va["dice_ED"] + va["dice_ET"]) / 3.0
            print(f"  [P{phase} ep {e+1}/{n_ep}] alpha={va.get('alpha', float('nan')):.2f}  "
                  f"{format_metrics(va, 'val')}  fg_dice={fg:.4f}")
            if fg > best_fg:
                best_fg, best_metrics = fg, va
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, opt, sched, g, va)
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, opt, sched, g, va)
            g += 1

    _report_vram(device)
    print(f"  → best FG Dice GSL ({arch}): {best_fg:.4f}")
    if backup_dir:
        _backup(ckpt_dir, backup_dir, arch)
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


def _backup(ckpt_dir: str, backup_dir: str, arch: str) -> None:
    """Copia i checkpoint di un modello nella cartella di backup (es. Drive)."""
    dst = os.path.join(backup_dir, arch)
    os.makedirs(dst, exist_ok=True)
    for name in ("best.pth", "last.pth", "history.json"):
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    print(f"  [backup] {ckpt_dir} → {dst}")


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
    if args.epochs is not None:
        overrides["phase2_epochs"] = args.epochs
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

    # 4. Training
    if args.train:
        results = {}
        for arch in args.models:
            results[arch] = train_model(
                arch, cfg, device, args.data_root, args.ckpt_root,
                use_gsl=args.gsl, backup_dir=args.backup_dir,
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
        run_evaluation(args.models, args.data_root, args.ckpt_root)
        if args.backup_dir:
            out = PROJECT_ROOT / "evaluation_outputs"
            if out.is_dir():
                dst = os.path.join(args.backup_dir, "evaluation_outputs")
                shutil.copytree(out, dst, dirs_exist_ok=True)
                print(f"  [backup] evaluation_outputs → {dst}")
    else:
        print("\n[skip] valutazione disabilitata (--no-eval).")

    print(f"\n[DONE] Pipeline completata in {(time.time()-t_start)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
