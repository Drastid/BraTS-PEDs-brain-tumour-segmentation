#!/usr/bin/env python
"""
run_ensemble_3d.py
=====================
Entrypoint per la valutazione dell'ENSEMBLE SegResNet ⊕ SwinUNETR (roadmap
Sezione 2, Fase C/D). NON allena nulla: carica i due `best.pth` gia' prodotti
da `run_pipeline_3d.py --all` (o due run separate con lo stesso `--loss`),
valuta l'ensemble sul test set (src/ensemble_3d.py — media delle probabilita'
softmax, DEC-1/DEC-2), poi costruisce il confronto a TRE vie (SegResNet,
SwinUNETR, Ensemble) leggendo i `test_3d_metrics.json` gia' salvati dalle
run singole, con test di significativita' Wilcoxon (DEC-4) e verdetto di
accettazione (DEC-5: dice_mean_fg migliora E p<0.05).

Precondizione: entrambe le architetture gia' allenate e valutate con lo
STESSO --run-name e --loss (default dice_focal — DEC-3: perimetro solo
dice_focal, l'ensemble su --loss gsl e' rimandato a dopo aver validato
l'ensemble di base), cosi' che `evaluation_outputs/<run-name>/segresnet_<loss>/
test_3d_metrics.json` e l'equivalente per swinunetr esistano gia'.

Uso:
    python run_ensemble_3d.py \\
        --data-root data/processed_3d \\
        --run-name run01 \\
        --loss dice_focal \\
        --segresnet-ckpt checkpoints/run01/segresnet_dice_focal/best.pth \\
        --swinunetr-ckpt checkpoints/run01/swinunetr_dice_focal/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FG_CLASSES = ("ET", "NET", "CC", "ED")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Valutazione ensemble SegResNet+SwinUNETR (media softmax) + confronto a 3 vie.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="Root con la sottocartella 'test' (es. data/processed_3d).")
    p.add_argument("--segresnet-ckpt", default=None,
                   help="Path al best.pth di SegResNet. Se omesso, SegResNet e' escluso dall'ensemble.")
    p.add_argument("--swinunetr-ckpt", default=None,
                   help="Path al best.pth di SwinUNETR. Se omesso, SwinUNETR e' escluso dall'ensemble.")
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128],
                   help="Dimensione della finestra scorrevole (deve combaciare con quella di training).")
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--run-name", default="run01",
                   help="Deve combaciare col --run-name usato per allenare/valutare le due arch singole "
                        "(serve per ritrovare i loro test_3d_metrics.json).")
    p.add_argument("--loss", default="dice_focal",
                   help="Deve combaciare col --loss delle due run singole (default dice_focal — DEC-3: "
                        "perimetro dell'ensemble limitato a dice_focal).")
    p.add_argument("--eval-out-root", default="evaluation_outputs")
    p.add_argument("--no-postprocessing", dest="postprocessing", action="store_false",
                   help="Disattiva remove_small_components sulla predizione ensemble.")
    p.set_defaults(postprocessing=True)
    p.add_argument("--eval-min-component-voxels", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Soglia di significativita' per il test di Wilcoxon (DEC-4/DEC-5).")
    return p.parse_args(argv)


def _load_single_model_metrics(eval_out_root: str, run_name: str, arch: str, loss: str) -> Optional[dict]:
    """Legge test_3d_metrics.json di una run singola gia' valutata, se presente."""
    path = os.path.join(eval_out_root, run_name, f"{arch}_{loss}", "test_3d_metrics.json")
    if not os.path.isfile(path):
        print(f"[warn] {path} non trovato: {arch} escluso dal confronto a 3 vie "
              f"(assicurati di aver gia' valutato {arch} con --run-name={run_name!r} --loss={loss!r}).")
        return None
    with open(path) as f:
        return json.load(f)


def _print_class_table(label: str, summary: dict) -> None:
    print(f"\n--- {label} ---")
    for cls in FG_CLASSES:
        d_mean = summary.get(f"dice_{cls}_mean", float("nan"))
        d_std = summary.get(f"dice_{cls}_std", float("nan"))
        h_mean = summary.get(f"hd95_{cls}_mean", float("nan"))
        h_std = summary.get(f"hd95_{cls}_std", float("nan"))
        marker = "  <-- CC" if cls == "CC" else ""
        print(f"  {cls:4s}  dice={d_mean:.4f}±{d_std:.4f}   hd95={h_mean:6.2f}±{h_std:5.2f}{marker}")
    print(f"  {'mean_fg':4s}  dice={summary.get('dice_mean_fg_mean', float('nan')):.4f}±"
          f"{summary.get('dice_mean_fg_std', float('nan')):.4f}   "
          f"hd95={summary.get('hd95_mean_fg_mean', float('nan')):6.2f}±"
          f"{summary.get('hd95_mean_fg_std', float('nan')):5.2f}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.segresnet_ckpt and not args.swinunetr_ckpt:
        print("[ERRORE] serve almeno un checkpoint tra --segresnet-ckpt e --swinunetr-ckpt.")
        return 1

    from src.ensemble_3d import (
        compare_ensemble_vs_best_single,
        ensemble_acceptance_verdict,
        evaluate_test_set_3d_ensemble,
    )
    from src.eval_3d import summarize_metrics
    from src.models3d import build_model_3d
    from src.train_3d import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device={device}")
    roi = tuple(args.roi)

    # --- Fase C2: carica i modelli gia' allenati ---
    models = []
    arch_names = []
    for arch, ckpt_path in (("segresnet", args.segresnet_ckpt), ("swinunetr", args.swinunetr_ckpt)):
        if not ckpt_path:
            continue
        if not os.path.isfile(ckpt_path):
            print(f"[ERRORE] checkpoint non trovato per {arch}: {ckpt_path}")
            return 1
        model = build_model_3d(arch, in_channels=4, num_classes=args.num_classes, roi=roi).to(device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()
        models.append(model)
        arch_names.append(arch)
        print(f"  caricato {arch} da {ckpt_path}")
    print(f"\nEnsemble di {len(models)} modelli: {'+'.join(arch_names)}")

    # --- Fase D1: valutazione ensemble sul test set ---
    test_dir = os.path.join(args.data_root, "test")
    out_dir = os.path.join(args.eval_out_root, args.run_name, "ensemble")
    nifti_dir = os.path.join(out_dir, "nifti_predictions")

    print(f"\n{'='*70}\n  VALUTAZIONE ENSEMBLE — test set ({'+'.join(arch_names)}/{args.loss})\n{'='*70}")
    per_subject_ensemble = evaluate_test_set_3d_ensemble(
        models, test_dir, device, roi=roi, num_classes=args.num_classes,
        apply_postprocessing=args.postprocessing,
        min_component_voxels=args.eval_min_component_voxels,
        export_nifti_dir=nifti_dir, raw_data_dir=test_dir,
    )
    ensemble_summary = summarize_metrics(per_subject_ensemble)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_3d_metrics.json"), "w") as f:
        json.dump({"per_subject": per_subject_ensemble, "summary": ensemble_summary}, f, indent=2)
    print(f"  [eval] risultati ensemble -> {out_dir}")

    # --- Fase B2/D2: tabella a 3 vie, leggendo le run singole gia' valutate ---
    single_results = {}
    for arch in ("segresnet", "swinunetr"):
        data = _load_single_model_metrics(args.eval_out_root, args.run_name, arch, args.loss)
        if data is not None:
            single_results[arch] = data

    print(f"\n{'='*70}\n  CONFRONTO A 3 VIE — test set\n{'='*70}")
    for arch, data in single_results.items():
        _print_class_table(f"{arch} / {args.loss}", data["summary"])
    _print_class_table(f"ensemble ({'+'.join(arch_names)}) / {args.loss}", ensemble_summary)

    # --- Fase D2: Wilcoxon vs il MIGLIOR modello singolo disponibile ---
    if not single_results:
        print("\n[warn] nessun test_3d_metrics.json single-model trovato: "
              "salto il confronto statistico (DEC-4/DEC-5). Valuta prima "
              "le due architetture singolarmente con --evaluate (default ON "
              "in run_pipeline_3d.py).")
        return 0

    best_arch = max(
        single_results, key=lambda a: single_results[a]["summary"].get("dice_mean_fg_mean", float("-inf"))
    )
    best_dice = single_results[best_arch]["summary"].get("dice_mean_fg_mean", float("nan"))
    print(f"\nMiglior modello singolo: {best_arch} (dice_mean_fg={best_dice:.4f})")

    comparison = compare_ensemble_vs_best_single(
        per_subject_ensemble, single_results[best_arch]["per_subject"], metric_key="dice_mean_fg",
    )
    verdict = ensemble_acceptance_verdict(comparison, alpha=args.alpha)

    print(f"\n--- Test statistico (Wilcoxon signed-rank, DEC-4) ---")
    print(f"  n_subjects       = {comparison['n_subjects']}")
    print(f"  mean ensemble    = {comparison['mean_ensemble']:.4f}")
    print(f"  mean {best_arch:10s} = {comparison['mean_best_single']:.4f}")
    print(f"  mean_diff        = {comparison['mean_diff']:+.4f}")
    print(f"  wilcoxon stat    = {comparison['wilcoxon_stat']:.4f}")
    print(f"  p_value          = {comparison['p_value']:.4f}  (alpha={args.alpha})")

    verdict_str = "ACCETTATO" if verdict["accepted"] else "NON accettato"
    print(f"\n--- Verdetto (DEC-5: dice_mean_fg migliora E p<{args.alpha}) ---")
    print(f"  Ensemble {verdict_str}: improved={verdict['improved']}  significant={verdict['significant']}")

    with open(os.path.join(out_dir, "ensemble_comparison.json"), "w") as f:
        json.dump({"best_single_arch": best_arch, "verdict": verdict}, f, indent=2)
    print(f"  [eval] verdetto salvato -> {os.path.join(out_dir, 'ensemble_comparison.json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
