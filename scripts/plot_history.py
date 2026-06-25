#!/usr/bin/env python
"""
scripts/plot_history.py
=======================
Curve train-vs-val per la verifica di overfitting (raccomandazione "no overfit").

Legge i file ``<ckpt_root>/<arch>/history.json`` prodotti da run_pipeline e
disegna, per ogni modello, due pannelli: loss (train/val) e foreground Dice
(train/val), con una linea verticale sul confine Phase 1 -> Phase 2.

ESEMPI
------
    python -m scripts.plot_history --ckpt-root /content/checkpoints/baseline
    python -m scripts.plot_history --ckpt-root /content/checkpoints/gsl \
        --models unet --out-dir /content/drive/MyDrive/Brats_Project/checkpoints/gsl

Output: ``<out-dir>/history_<arch>.png`` (out-dir = ckpt-root se non specificato).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva su file
import matplotlib.pyplot as plt


def _plot_one(history_path: str, out_path: str) -> None:
    with open(history_path) as f:
        hist = json.load(f)
    if not hist:
        print(f"[skip] {history_path} vuoto")
        return

    ep = [r["epoch"] for r in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ep, [r["train_loss"] for r in hist], marker="o", ms=3, label="train")
    ax1.plot(ep, [r["val_loss"] for r in hist], marker="o", ms=3, label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoca"); ax1.set_ylabel("loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [r["train_fg_dice"] for r in hist], marker="o", ms=3, label="train")
    ax2.plot(ep, [r["val_fg_dice"] for r in hist], marker="o", ms=3, label="val")
    ax2.set_title("Foreground Dice  (NCR+ED+ET)/3")
    ax2.set_xlabel("epoca"); ax2.set_ylabel("Dice")
    ax2.legend(); ax2.grid(alpha=0.3)

    # confine Phase 1 -> Phase 2 (encoder sbloccato)
    p2 = [r["epoch"] for r in hist if r.get("phase") == 2]
    if p2:
        for ax in (ax1, ax2):
            ax.axvline(p2[0] - 0.5, ls="--", c="grey", alpha=0.6)

    arch = os.path.basename(os.path.dirname(history_path)) or "model"
    fig.suptitle(f"{arch} — curve train/val")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    last = hist[-1]
    gap = (last.get("train_fg_dice") or 0.0) - (last.get("val_fg_dice") or 0.0)
    print(f"salvato: {out_path}")
    print(f"  ultimo epoch: train_fg={last.get('train_fg_dice'):.3f}  "
          f"val_fg={last.get('val_fg_dice'):.3f}  gap(train-val)={gap:+.3f}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot delle curve train/val da history.json (verifica overfitting).")
    ap.add_argument("--ckpt-root", required=True,
                    help="Cartella che contiene <arch>/history.json.")
    ap.add_argument("--models", nargs="+", default=["unet", "fpn", "segformer"],
                    help="Modelli da plottare (default: tutti e tre).")
    ap.add_argument("--out-dir", default=None,
                    help="Dove salvare i PNG (default: --ckpt-root).")
    a = ap.parse_args(argv)

    out_dir = a.out_dir or a.ckpt_root
    os.makedirs(out_dir, exist_ok=True)

    found = 0
    for m in a.models:
        hp = os.path.join(a.ckpt_root, m, "history.json")
        if not os.path.isfile(hp):
            print(f"[skip] {hp} non trovato")
            continue
        _plot_one(hp, os.path.join(out_dir, f"history_{m}.png"))
        found += 1
    if found == 0:
        print("Nessun history.json trovato: allena prima con run_pipeline aggiornato.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
