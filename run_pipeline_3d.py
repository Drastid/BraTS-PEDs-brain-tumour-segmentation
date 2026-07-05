#!/usr/bin/env python
"""
run_pipeline_3d.py
====================
Orchestratore end-to-end per il training 3D BraTS-PEDs su Colab/A100.

    python run_pipeline_3d.py --arch segresnet --loss dice_focal --data-root /content/processed_3d
    python run_pipeline_3d.py --arch swinunetr --loss gsl --pretrained none
    python run_pipeline_3d.py --all --loss dice_focal \
        --pretrained-swinunetr weights/model_swinvit.pt \
        --pretrained-segresnet weights/brats_mri_segmentation.pt

Il flag --loss (richiesto esplicitamente dall'utente nel Punto 4 della
roadmap) seleziona il ramo di training:
    --loss dice_focal : monai.losses.DiceFocalLoss (src/losses_monai.py) —
                        pipeline di default, meno rischio.
    --loss gsl        : DiceFocalGSLLoss schedulata (src/losses_3d.py) con
                        DTM calcolata on-the-fly per patch
                        (src/transforms.py::ComputeDistanceMapd).

Il flag --pretrained {auto,none} (road_3D.md §7) permette di disattivare il
transfer learning per un'ablation "scratch vs pretrained". I checkpoint sono
specifici per architettura — --pretrained-swinunetr e --pretrained-segresnet
(nessun --weights-path condiviso: un checkpoint SwinUNETR non ha alcuna
chiave in comune con SegResNet/DynUNet). DynUNet non ha un flag pesi
dedicato: non e' stato individuato alcun checkpoint pre-addestrato compatibile
per questa architettura nel progetto, quindi parte sempre da zero, sia con
--arch dynunet sia dentro --all.

Backup e valutazione (road_3D.md §7, completato al Punto 7)
-------------------------------------------------------------
--backup-dir DIR   Dopo il training, copia best.pth/last.pth/history.json in
                   DIR/<run-name>/<arch>_<loss>/ (es. una cartella su Google
                   Drive montata su Colab). Essenziale su Colab: il runtime è
                   effimero e i checkpoint locali si perdono a disconnessione.
--evaluate / --no-evaluate  Dopo il training (default: ON), esegue
                   automaticamente la valutazione 3D nativa sul test set
                   (src/eval_3d.py, Punto 6: Dice/HD95 per-classe +
                   post-processing + export NIfTI) e salva i risultati in
                   evaluation_outputs/<run-name>/<arch>_<loss>/.

Questo script allena un modello/loss per invocazione (--arch singolo), oppure
tutti e tre i modelli in sequenza con --all (stessa logica di
master/run_pipeline.py --models unet fpn segformer nel vecchio progetto 2D):
in quel caso --arch e' ignorato e si itera su ARCH_NAMES
(dynunet, segresnet, swinunetr), ciascuno con la propria sottocartella di
checkpoint/valutazione (<ckpt-root>/<run-name>/<arch>_<loss>/), cosi' i run
restano isolati e confrontabili tra loro.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Training 3D BraTS-PEDs (MONAI, Colab/A100).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arch", required=False, default=None,
                   choices=["dynunet", "segresnet", "swinunetr"],
                   help="Architettura da allenare (richiesto se --all non e' passato).")
    p.add_argument("--all", dest="all_archs", action="store_true",
                   help="Allena in sequenza tutte e tre le architetture "
                        "(dynunet, segresnet, swinunetr), ignorando --arch. "
                        "Ciascuna usa la propria sottocartella di checkpoint/valutazione "
                        "(stessa logica di master/run_pipeline.py --models nel 2D).")
    p.add_argument("--loss", default="dice_focal", choices=["dice_focal", "gsl"],
                   help="Ramo di loss: 'dice_focal' (MONAI, default sicuro) o 'gsl' (schedulata).")

    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "processed_3d"),
                   help="Root con le sottocartelle train/val (NIfTI riorganizzati).")
    p.add_argument("--ckpt-root", default=str(PROJECT_ROOT / "checkpoints"),
                   help="Dove salvare i checkpoint.")

    p.add_argument("--pretrained", default="auto", choices=["auto", "none"],
                   help="'auto' carica i pesi pre-addestrati dal flag --pretrained-<arch> "
                        "corrispondente (se fornito); 'none' allena sempre da zero (ablation).")
    p.add_argument("--pretrained-swinunetr", default=None,
                   help="Path al checkpoint pre-addestrato per SwinUNETR (es. pesi SSL NVIDIA "
                        "model_swinvit.pt, o fine-tuned HuggingFace/BrainSegFounder "
                        "model_best_fold_0.pth). Ignorato se --pretrained=none o se l'arch in "
                        "esecuzione non e' swinunetr.")
    p.add_argument("--pretrained-segresnet", default=None,
                   help="Path al checkpoint pre-addestrato per SegResNet (es. bundle MONAI "
                        "Model Zoo brats_mri_segmentation). Ignorato se --pretrained=none o se "
                        "l'arch in esecuzione non e' segresnet.")
    # DynUNet non ha un flag pesi dedicato: nessun checkpoint pre-addestrato
    # compatibile e' stato individuato per questa architettura nel progetto
    # (road_3D.md §3.4) — parte sempre da zero, con o senza --all.

    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128],
                   help="Dimensione della patch 3D (deve rispettare i vincoli di "
                        "src/models3d.py: divisibile per 32 per SwinUNETR).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=2,
                   help="Patch campionate per volume ad ogni draw di training.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--warmup-freeze-epochs", type=int, default=0,
                   help="Se >0, congela il backbone per le prime N epoche (warm-up head-only, "
                        "consigliato per SwinUNETR — road_3D.md §5.3).")
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None,
                   help="Se impostato, checkpoint in <ckpt-root>/<run-name>/<arch>_<loss>/.")

    p.add_argument("--early-stopping", dest="early_stopping", action="store_true",
                   help="Interrompe il training se dice_mean_fg (validazione) non migliora "
                        "per --es-patience epoche consecutive (porting di "
                        "master/run_pipeline.py::_EarlyStopper). Default: OFF (comportamento "
                        "identico a prima — il training corre sempre per --epochs epoche).")
    p.set_defaults(early_stopping=False)
    p.add_argument("--es-patience", type=int, default=15,
                   help="Epoche consecutive senza miglioramento tollerate prima di fermarsi "
                        "(ignorato se --early-stopping non e' passato).")
    p.add_argument("--es-min-delta", type=float, default=1e-4,
                   help="Incremento minimo di dice_mean_fg per contare come miglioramento.")
    p.add_argument("--es-smooth-window", type=int, default=1,
                   help="Media mobile di dice_mean_fg su N epoche prima di valutare "
                        "l'early stopping (1 = nessuno smoothing, valore grezzo dell'epoca).")

    p.add_argument("--backup-dir", default=None,
                   help="Cartella (es. su Google Drive montato) dove copiare best.pth/"
                        "last.pth/history.json dopo il training. Essenziale su Colab: il "
                        "runtime e' effimero e i checkpoint locali si perdono a disconnessione.")

    p.add_argument("--evaluate", dest="evaluate", action="store_true",
                   help="Esegui la valutazione 3D nativa sul test set dopo il training (default).")
    p.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                   help="Salta la valutazione 3D dopo il training.")
    p.set_defaults(evaluate=True)
    p.add_argument("--eval-out-root", default=str(PROJECT_ROOT / "evaluation_outputs"),
                   help="Root dove salvare metriche/NIfTI della valutazione 3D.")
    p.add_argument("--eval-min-component-voxels", type=int, default=50,
                   help="Soglia (voxel) per remove_small_components in valutazione.")
    p.add_argument("--eval-no-postprocessing", dest="eval_postprocessing",
                   action="store_false",
                   help="Disattiva remove_small_components in valutazione (default: attivo).")
    p.set_defaults(eval_postprocessing=True)

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.all_archs and not args.arch:
        print("[ERRORE] specifica --arch <dynunet|segresnet|swinunetr> oppure --all.")
        return 1
    if args.all_archs and args.arch:
        print("[ERRORE] --arch e --all sono mutuamente esclusivi (--all itera su tutte "
              "le architetture, --arch ne seleziona una sola).")
        return 1

    import torch

    from src.models3d import ARCH_NAMES
    from src.train_3d import set_seed

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device={device} | "
          f"GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else '-'}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    archs = list(ARCH_NAMES) if args.all_archs else [args.arch]
    results: dict[str, float] = {}
    for arch in archs:
        results[arch] = _train_one_arch(args, arch, device)

    if len(archs) > 1:
        print(f"\n{'='*70}\n  RIEPILOGO best dice_mean_fg (val) — tutte le architetture\n{'='*70}")
        for arch, fg in results.items():
            print(f"  {arch:12s}  dice_mean_fg={fg:.4f}")

    return 0


def _train_one_arch(args: argparse.Namespace, arch: str, device) -> float:
    """Allena e valuta UNA architettura (args.arch e' ignorato: si usa `arch`).

    Estratta da main() per supportare --all (roadmap: training sequenziale di
    tutte e tre le architetture, stessa logica di
    master/run_pipeline.py::train_model iterato su --models nel 2D). Ogni
    chiamata usa la propria sottocartella <ckpt-root>/<run-name>/<arch>_<loss>/,
    cosi' i checkpoint non si sovrascrivono tra un'architettura e l'altra.

    Returns:
        best_fg_dice: il miglior dice_mean_fg di validazione visto durante il
        training di questa architettura (per il riepilogo finale di main()).
    """
    import torch

    from src.dataset_3d import build_dataloaders_3d
    from src.losses_3d import AlphaScheduler, DiceFocalGSLLoss
    from src.losses_monai import build_dice_focal_loss
    from src.models3d import build_model_3d, load_pretrained_3d
    from src.optim_3d import build_optimizer_3d, set_backbone_trainable
    from src.train_3d import (
        EarlyStopper3D, evaluate_3d, monitor_value, save_checkpoint,
        train_one_epoch_3d, train_one_epoch_gsl_3d,
    )

    roi = tuple(args.roi)
    run_label = args.run_name if args.run_name else "default"
    ckpt_dir = os.path.join(args.ckpt_root, run_label, f"{arch}_{args.loss}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Modello + (opzionale) pesi pre-addestrati ---
    # Il flag pesi e' specifico per architettura (--pretrained-swinunetr /
    # --pretrained-segresnet): un checkpoint SwinUNETR (es. model_swinvit.pt)
    # non ha alcuna chiave in comune con SegResNet/DynUNet, quindi non ha senso
    # un --weights-path condiviso — soprattutto con --all, dove servirebbe
    # altrimenti scegliere UN SOLO path per tutti e tre i modelli. DynUNet non
    # ha un flag dedicato: parte sempre da zero (nessun checkpoint compatibile
    # individuato per questa architettura, road_3D.md §3.4).
    weights_path_by_arch = {
        "swinunetr": args.pretrained_swinunetr,
        "segresnet": args.pretrained_segresnet,
    }
    weights_path = weights_path_by_arch.get(arch)

    model = build_model_3d(arch, in_channels=4, num_classes=args.num_classes, roi=roi).to(device)
    unmatched_param_names: list[str] = []
    if args.pretrained == "auto" and weights_path:
        model, unmatched_param_names = load_pretrained_3d(model, weights_path)
    elif args.pretrained == "auto" and not weights_path:
        print(f"[warn] --pretrained=auto ma nessun checkpoint fornito per arch={arch!r}: "
              f"training da zero.")

    # --- DataLoader ---
    with_dtm = args.loss == "gsl"
    train_loader, val_loader = build_dataloaders_3d(
        data_root=args.data_root, roi=roi, num_classes=args.num_classes,
        batch_size=args.batch_size, num_samples=args.num_samples,
        num_workers=args.num_workers, with_dtm=with_dtm,
    )

    # --- Optimizer (LR differenziato backbone/head) ---
    # extra_head_param_names: parametri esclusi dal caricamento pre-addestrato
    # per shape mismatch FUORI dalla head strutturale (es. patch_embed a 1
    # canale nel checkpoint SSL NVIDIA vs le 4 modalita' richieste qui). Sono
    # inizializzati a caso quanto la head, quindi vanno trattati come head
    # (LR pieno, mai congelati) — si veda src.optim_3d.split_backbone_head_params.
    optimizer = build_optimizer_3d(
        model, arch, base_lr=args.base_lr,
        backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay,
        extra_head_param_names=unmatched_param_names,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Loss ---
    if args.loss == "dice_focal":
        criterion = build_dice_focal_loss(num_classes=args.num_classes).to(device)
    else:  # gsl
        gsl_weights_path = os.path.join(args.data_root, "gsl_class_weights.json")
        gsl_weights = None
        if os.path.isfile(gsl_weights_path):
            with open(gsl_weights_path) as f:
                gsl_weights = json.load(f)["weights"]
        else:
            print(f"[warn] {gsl_weights_path} non trovato: la GSL userà pesi uniformi.")
        alpha_scheduler = AlphaScheduler(schedule="step", total_epochs=args.epochs, step_length=5)
        criterion = DiceFocalGSLLoss(
            num_classes=args.num_classes, gsl_class_weights=gsl_weights, scheduler=alpha_scheduler,
        ).to(device)

    # --- Warm-up opzionale a backbone congelato ---
    # extra_head_param_names qui sotto assicura che i parametri non
    # pre-addestrati (shape mismatch fuori dalla head, es. patch_embed) NON
    # vengano congelati insieme al backbone: restano allenabili durante tutto
    # il warm-up, esattamente come la head.
    if args.warmup_freeze_epochs > 0:
        set_backbone_trainable(
            model, arch, trainable=False, extra_head_param_names=unmatched_param_names,
        )

    print(f"\n{'#'*70}\n  TRAIN 3D  arch={arch}  loss={args.loss}  "
          f"roi={roi}  batch={args.batch_size}  epochs={args.epochs}\n{'#'*70}")

    stopper = EarlyStopper3D(args.early_stopping, args.es_patience, args.es_min_delta)
    if args.early_stopping:
        _mon = f"dice_mean_fg (media mobile {args.es_smooth_window})" if args.es_smooth_window > 1 else "dice_mean_fg"
        print(f"  [early-stopping] ON — monitor={_mon} patience={args.es_patience} "
              f"min_delta={args.es_min_delta}")

    best_fg_dice = -1.0
    history: list[dict] = []
    for epoch in range(args.epochs):
        if args.warmup_freeze_epochs > 0 and epoch == args.warmup_freeze_epochs:
            set_backbone_trainable(
                model, arch, trainable=True, extra_head_param_names=unmatched_param_names,
            )

        if args.loss == "dice_focal":
            tr_metrics = train_one_epoch_3d(
                model, train_loader, criterion, optimizer, device, amp_dtype=args.amp_dtype,
            )
        else:
            tr_metrics = train_one_epoch_gsl_3d(
                model, train_loader, criterion, optimizer, device, epoch=epoch,
                amp_dtype=args.amp_dtype,
            )
        scheduler.step()

        val_metrics = evaluate_3d(
            model, val_loader, device, roi=roi, num_classes=args.num_classes,
        )
        fg_dice = val_metrics["dice_mean_fg"]

        print(f"  [ep {epoch+1}/{args.epochs}] train_loss={tr_metrics['loss']:.4f}  "
              f"val_dice_fg={fg_dice:.4f}  val_hd95_fg={val_metrics['hd95_mean_fg']:.2f}")

        history.append({"epoch": epoch, "train": tr_metrics, "val": val_metrics})

        if fg_dice > best_fg_dice:
            best_fg_dice = fg_dice
            save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, optimizer, epoch, val_metrics)
        save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, optimizer, epoch, val_metrics)

        # Early stopping: la selezione di best.pth sopra resta SEMPRE legata al
        # fg_dice grezzo dell'epoca (comportamento identico a prima); lo
        # stopper valuta invece la media mobile (monitor_value), cosi' un
        # singolo run rumoroso non fa scattare uno stop prematuro.
        mon = monitor_value(history, args.es_smooth_window)
        if stopper.update(mon, epoch):
            print(f"  [early-stopping] stop a epoch {epoch}: nessun miglioramento di "
                  f"dice_mean_fg per {args.es_patience} epoche "
                  f"(best={stopper.best:.4f} @ ep{stopper.best_epoch}).")
            break

    history_path = os.path.join(ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] best val dice_mean_fg ({arch}/{args.loss}): {best_fg_dice:.4f}")
    print(f"  [history] curve train/val -> {history_path} ({len(history)} epoche)")

    # --- Backup opzionale su cartella esterna (es. Google Drive su Colab) ---
    if args.backup_dir:
        _backup_checkpoints(ckpt_dir, args.backup_dir, run_label, arch, args.loss)

    # --- Valutazione 3D nativa automatica sul test set (default: ON) ---
    if args.evaluate:
        _run_evaluation(
            model=model, device=device, args=args, arch=arch, roi=roi,
            best_ckpt_path=os.path.join(ckpt_dir, "best.pth"), run_label=run_label,
        )

    return best_fg_dice


def _backup_checkpoints(
    ckpt_dir: str, backup_dir: str, run_label: str, arch: str, loss: str,
) -> None:
    """Copia best.pth/last.pth/history.json in backup_dir/<run_label>/<arch>_<loss>/.

    Essenziale su Colab (road_3D.md §7): il runtime e' effimero e i checkpoint
    salvati solo su disco locale della VM si perdono a disconnessione. Stessa
    logica di `master/run_pipeline.py::_backup` (namespacing per run_label,
    cosi' run diversi non si sovrascrivono a vicenda).
    """
    dst = os.path.join(backup_dir, run_label, f"{arch}_{loss}")
    os.makedirs(dst, exist_ok=True)
    for name in ("best.pth", "last.pth", "history.json"):
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    print(f"  [backup] {ckpt_dir} -> {dst}")


def _run_evaluation(model, device, args, arch: str, roi, best_ckpt_path: str, run_label: str) -> None:
    """Carica il best checkpoint e valuta sul test set (roadmap §6, Punto 6).

    Usa esattamente la pipeline di src/eval_3d.py: inferenza sliding-window
    nativa 3D, remove_small_components opzionale, Dice/HD95 per-classe, export
    NIfTI per OGNI soggetto del test set (decisione presa al Punto 6).

    `arch` e' passato esplicitamente (non args.arch) per supportare --all:
    ogni chiamata di questa funzione valuta l'architettura corrente del loop
    in main(), non necessariamente quella (eventualmente assente) in args.
    """
    import json as _json

    from src.eval_3d import evaluate_test_set_3d, summarize_metrics
    from src.train_3d import load_checkpoint

    if not os.path.isfile(best_ckpt_path):
        print(f"[warn] {best_ckpt_path} non trovato: valutazione saltata.")
        return

    load_checkpoint(best_ckpt_path, model, device=device)
    model.eval()

    test_dir = os.path.join(args.data_root, "test")
    out_dir = os.path.join(args.eval_out_root, run_label, f"{arch}_{args.loss}")
    nifti_dir = os.path.join(out_dir, "nifti_predictions")

    print(f"\n{'='*70}\n  VALUTAZIONE 3D NATIVA — test set ({arch}/{args.loss})\n{'='*70}")
    per_subject = evaluate_test_set_3d(
        model, test_dir, device, roi=roi, num_classes=args.num_classes,
        apply_postprocessing=args.eval_postprocessing,
        min_component_voxels=args.eval_min_component_voxels,
        export_nifti_dir=nifti_dir, raw_data_dir=test_dir,
    )
    summary = summarize_metrics(per_subject)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_3d_metrics.json"), "w") as f:
        _json.dump({"per_subject": per_subject, "summary": summary}, f, indent=2)

    print(f"  dice_mean_fg: {summary.get('dice_mean_fg_mean', float('nan')):.4f} "
          f"± {summary.get('dice_mean_fg_std', float('nan')):.4f}")
    print(f"  hd95_mean_fg: {summary.get('hd95_mean_fg_mean', float('nan')):.2f} "
          f"± {summary.get('hd95_mean_fg_std', float('nan')):.2f}")
    print(f"  [eval] risultati -> {out_dir}")

    if args.backup_dir:
        dst = os.path.join(args.backup_dir, "evaluation_outputs", run_label, f"{arch}_{args.loss}")
        shutil.copytree(out_dir, dst, dirs_exist_ok=True)
        print(f"  [backup] {out_dir} -> {dst}")


if __name__ == "__main__":
    raise SystemExit(main())
