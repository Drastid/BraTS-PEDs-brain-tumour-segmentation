"""
src/ensemble_3d.py
=====================
Ensemble di piu' modelli 3D gia' allenati SEPARATAMENTE (roadmap Sezione 2:
Ensemble Strategy SegResNet ⊕ SwinUNETR).

Razionale (§2.0 della roadmap): SegResNet (CNN) e SwinUNETR (transformer) hanno
bias induttivi diversi — la CNN e' forte su feature locali/texture, il
transformer sul contesto globale a lungo raggio. I loro errori sono quindi
parzialmente decorrelati, la condizione che rende un ensemble utile.

Decisioni tecniche adottate (§2.6 della roadmap):
    DEC-1 (fusione):      Opzione A — media semplice delle probabilita'
                          (soft voting), nessun iperparametro. `weights=None`
                          nelle funzioni sotto lascia aperta l'Opzione B
                          (media pesata) per un domani, senza rompere l'API.
    DEC-2 (livello):       Opzione A — fusione a livello di probabilita'
                          POST-softmax (non sui logit grezzi: SegResNet e
                          SwinUNETR possono avere logit di scala/calibrazione
                          diversa; non sulle label: con 2 soli modelli il
                          voto di maggioranza degenera nei pareggi).
    DEC-4 (significativita'): Opzione A — Wilcoxon signed-rank sui Dice
                          per-soggetto (non parametrico, adatto a campioni
                          piccoli/non-normali come i ~26 soggetti test).
    DEC-5 (accettazione):  Opzione A — dice_mean_fg migliora E p<0.05.

Riusa deliberatamente predict_probs_3d/remove_small_components/
SegmentationMetrics3D/summarize_metrics di src/eval_3d.py e src/metrics_3d.py:
l'ensemble non duplica la logica di inferenza/metrica, solo la fonde.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .constants import NUM_CLASSES
from .eval_3d import (
    get_subject_ids,
    load_subject_affine,
    predict_probs_3d,
    remove_small_components,
    save_prediction_nifti,
)
from .metrics_3d import SegmentationMetrics3D

# ---------------------------------------------------------------------------
# Fusione delle probabilita' (DEC-1)
# ---------------------------------------------------------------------------


def average_probabilities(
    probs_list: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Fonde le probabilita' softmax di piu' modelli (DEC-1).

    Args:
        probs_list: Lista di tensori [C, *spatial] (stessa shape, stesso
                    ordine di classe), tipicamente l'output di
                    predict_probs_3d per ciascun modello sullo stesso soggetto.
        weights:    Se None (default, Opzione A — media semplice/soft voting),
                    peso uniforme 1/N per ciascun modello. Se fornito
                    (Opzione B — media pesata), deve avere la stessa
                    lunghezza di probs_list; viene normalizzato a somma 1
                    internamente (cosi' il chiamante puo' passare pesi non
                    normalizzati, es. Dice relativi grezzi).

    Returns:
        Tensore [C, *spatial]: media (pesata o uniforme) delle probabilita' —
        NON ri-normalizzato con un softmax aggiuntivo (la media di distribuzioni
        di probabilita' valide e' gia' una distribuzione di probabilita' valida,
        somma-a-1 per costruzione, se i pesi sommano a 1).
    """
    if not probs_list:
        raise ValueError("average_probabilities richiede almeno un tensore di probabilita'.")

    n = len(probs_list)
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"weights deve avere lunghezza {n} (una per modello), got {len(weights)}."
            )
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights deve sommare a un valore positivo.")
        w = [wi / total for wi in weights]

    fused = probs_list[0] * w[0]
    for probs, wi in zip(probs_list[1:], w[1:]):
        fused = fused + probs * wi
    return fused


# ---------------------------------------------------------------------------
# Inferenza ensemble per soggetto / sul test set
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_volume_3d_ensemble(
    models: Sequence[nn.Module],
    data_dir: str,
    subject_id: str,
    device: torch.device,
    roi: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    apply_postprocessing: bool = True,
    min_component_voxels: int = 50,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inferenza ENSEMBLE per un soggetto: sliding-window per ciascun modello
    (predict_probs_3d), fusione delle probabilita' (average_probabilities),
    poi argmax e SOLO A QUEL PUNTO remove_small_components (il post-processing
    si applica dopo la fusione, sull'argmax finale — §2.4 della roadmap: fondere
    prima, ripulire dopo, mai il contrario).

    Args:
        models: Lista di modelli 3D (>=1) gia' in eval mode, sullo stesso
                device — tipicamente [segresnet, swinunetr], ma funziona con
                qualunque numero/combinazione di modelli che condividano lo
                spazio di classi.
        Altri argomenti: identici a predict_volume_3d (src/eval_3d.py), con
        l'aggiunta di `weights` (vedi average_probabilities).

    Returns:
        (pred, pred_raw, gt): stessa tupla di predict_volume_3d.
    """
    if not models:
        raise ValueError("predict_volume_3d_ensemble richiede almeno un modello.")

    probs_list: List[torch.Tensor] = []
    gt: Optional[np.ndarray] = None
    for model in models:
        probs, gt_i = predict_probs_3d(
            model, data_dir, subject_id, device, roi=roi, num_classes=num_classes,
            sw_batch_size=sw_batch_size, overlap=overlap,
        )
        probs_list.append(probs)
        gt = gt_i  # identico per ogni modello (stesso soggetto, stessa GT)

    fused = average_probabilities(probs_list, weights=weights)
    pred_raw = fused.argmax(dim=0).numpy().astype(np.int16)  # [H, W, D]

    pred = pred_raw
    if apply_postprocessing:
        pred = remove_small_components(pred_raw, num_classes=num_classes, min_voxels=min_component_voxels)

    assert gt is not None  # per il type checker: il loop sopra gira almeno una volta
    return pred, pred_raw, gt


def evaluate_test_set_3d_ensemble(
    models: Sequence[nn.Module],
    data_dir: str,
    device: torch.device,
    roi: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    apply_postprocessing: bool = True,
    min_component_voxels: int = 50,
    export_nifti_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Valuta l'ENSEMBLE sull'intero test set.

    Stesso schema di ritorno di evaluate_test_set_3d (src/eval_3d.py): dict
    {subject_id: {dice_ET: ..., hd95_ET: ..., ..., dice_mean_fg, hd95_mean_fg}},
    per-soggetto, nessuna aggregazione qui — cosi' summarize_metrics() e
    compare_ensemble_vs_best_single() (sotto) funzionano senza modifiche, sia
    per un singolo modello sia per l'ensemble.
    """
    if export_nifti_dir is not None and raw_data_dir is None:
        raise ValueError("raw_data_dir e' richiesto quando export_nifti_dir e' fornito.")

    subject_ids = get_subject_ids(data_dir)
    print(f"Valutazione ENSEMBLE ({len(models)} modelli) su {len(subject_ids)} soggetti test...")

    per_subject_metrics: Dict[str, Dict[str, float]] = {}

    for i, subject_id in enumerate(subject_ids):
        pred, pred_raw, gt = predict_volume_3d_ensemble(
            models, data_dir, subject_id, device, roi=roi, num_classes=num_classes,
            apply_postprocessing=apply_postprocessing, min_component_voxels=min_component_voxels,
            weights=weights,
        )

        metrics = SegmentationMetrics3D(num_classes=num_classes)
        pred_t = torch.from_numpy(pred).long().unsqueeze(0)
        gt_t = torch.from_numpy(gt).long().unsqueeze(0)
        # Stesso trucco "pseudo-logits" di evaluate_test_set_3d: la predizione
        # e' gia' un'etichetta (post-fusione+argmax), non un logit — la
        # one-hot scalata e' equivalente ad argmax dopo softmax per le metriche.
        pseudo_logits = torch.nn.functional.one_hot(pred_t, num_classes=num_classes).permute(
            0, 4, 1, 2, 3
        ).float() * 10.0
        metrics.update(pseudo_logits, gt_t)
        per_subject_metrics[subject_id] = metrics.aggregate_and_reset()

        if export_nifti_dir is not None:
            affine, header = load_subject_affine(raw_data_dir, subject_id)
            out_path = f"{export_nifti_dir}/{subject_id}_pred.nii.gz"
            save_prediction_nifti(pred, affine, header, out_path)

        print(f"  [{i+1}/{len(subject_ids)}] {subject_id}  "
              f"dice_fg={per_subject_metrics[subject_id]['dice_mean_fg']:.4f}", end="\r")

    print()
    return per_subject_metrics


# ---------------------------------------------------------------------------
# Validazione statistica dell'ensemble (§2.5, DEC-4/DEC-5)
# ---------------------------------------------------------------------------


def compare_ensemble_vs_best_single(
    per_subject_ensemble: Dict[str, Dict[str, float]],
    per_subject_best_single: Dict[str, Dict[str, float]],
    metric_key: str = "dice_mean_fg",
) -> Dict[str, float]:
    """Confronta ensemble vs miglior modello singolo con Wilcoxon signed-rank
    (DEC-4 Opzione A), sui valori PER-SOGGETTO di `metric_key`.

    Wilcoxon signed-rank invece di un t-test appaiato: non assume normalita'
    delle differenze, appropriato per il campione piccolo (~26 soggetti test)
    di BraTS-PEDs.

    Args:
        per_subject_ensemble:    Output di evaluate_test_set_3d_ensemble.
        per_subject_best_single: Output di evaluate_test_set_3d (src/eval_3d.py)
                                 per il MIGLIOR modello singolo (quello con
                                 dice_mean_fg piu' alto tra SegResNet/SwinUNETR
                                 — la scelta di "quale sia il migliore" e'
                                 responsabilita' del chiamante, tipicamente
                                 run_ensemble_3d.py).
        metric_key:              Chiave della metrica su cui testare (default
                                 dice_mean_fg, il criterio primario di DEC-5).

    Returns:
        Dict con n_subjects, mean_ensemble, mean_best_single, mean_diff,
        wilcoxon_stat, p_value. I soggetti con NaN in uno dei due lati (classe
        foreground assente in quel soggetto) sono esclusi dal test.
    """
    from scipy.stats import wilcoxon

    common = sorted(set(per_subject_ensemble) & set(per_subject_best_single))
    if not common:
        raise ValueError(
            "Nessun subject_id in comune tra ensemble e modello singolo: "
            "sono stati valutati sullo stesso test set?"
        )

    ens_vals = np.array([per_subject_ensemble[s][metric_key] for s in common], dtype=np.float64)
    single_vals = np.array([per_subject_best_single[s][metric_key] for s in common], dtype=np.float64)

    mask = ~(np.isnan(ens_vals) | np.isnan(single_vals))
    ens_vals, single_vals = ens_vals[mask], single_vals[mask]

    mean_ens = float(np.mean(ens_vals)) if len(ens_vals) else float("nan")
    mean_single = float(np.mean(single_vals)) if len(single_vals) else float("nan")

    diff = ens_vals - single_vals
    if len(diff) == 0:
        stat, p_value = float("nan"), float("nan")
    elif np.allclose(diff, 0.0):
        # scipy.stats.wilcoxon solleva ValueError se tutte le differenze sono
        # zero (nessuna informazione per il test) — trattato esplicitamente
        # come "nessuna differenza rilevabile", non un errore.
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = wilcoxon(ens_vals, single_vals)

    return {
        "metric": metric_key,
        "n_subjects": int(mask.sum()),
        "mean_ensemble": mean_ens,
        "mean_best_single": mean_single,
        "mean_diff": mean_ens - mean_single,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
    }


def ensemble_acceptance_verdict(
    comparison: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Criterio di accettazione dell'ensemble (DEC-5 Opzione A): adottato se
    `dice_mean_fg` migliora (mean_diff > 0) E la differenza e' statisticamente
    significativa (p < alpha secondo compare_ensemble_vs_best_single).

    Args:
        comparison: Output di compare_ensemble_vs_best_single.
        alpha:      Soglia di significativita' (default 0.05).

    Returns:
        Dict con "accepted" (bool), "improved" (bool), "significant" (bool),
        "alpha", piu' tutti i campi di `comparison` per un report autonomo.
    """
    improved = comparison["mean_diff"] > 0
    significant = comparison["p_value"] < alpha
    return {
        "accepted": improved and significant,
        "improved": improved,
        "significant": significant,
        "alpha": alpha,
        **comparison,
    }
