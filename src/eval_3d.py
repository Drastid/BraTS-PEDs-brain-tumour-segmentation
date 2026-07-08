"""
src/eval_3d.py
=================
Valutazione 3D nativa sul test set BraTS-PEDs (roadmap §6, fase Colab).

A differenza del vecchio progetto 2D (master/src/eval_utils.py::predict_volume,
che ricostruiva il volume 3D impilando predizioni slice-by-slice con center-crop
simmetrico), qui la predizione e' NATIVAMENTE 3D: sliding_window_inference
lavora direttamente sul volume intero, senza alcun crop/un-crop — il modello
vede (a rotazione, tramite finestre sovrapposte) l'intero volume.

Pipeline per soggetto:
    1. Carica + normalizza il volume intero (stessa ClipAndNormalizeNonZerod
       del training — src/transforms_3d.py).
    2. sliding_window_inference -> argmax -> etichette di classe.
    3. (opzionale, default ON) remove_small_components: rimuove componenti
       connesse 3D isolate sotto una soglia di voxel — stesso "confetti
       effect" cleanup del vecchio progetto, meno frequente in 3D ma non
       impossibile.
    4. Metriche Dice + HD95 per-classe (src/metrics_3d.py).
    5. Export della predizione come NIfTI (stesso affine/header del soggetto
       originale), per ispezione visiva in ITK-SNAP/3D Slicer.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset_3d import MODALITIES, build_eval_transforms
from .metrics_3d import FOREGROUND_CLASS_NAMES, SegmentationMetrics3D


# ---------------------------------------------------------------------------
# Post-processing: rimozione componenti connesse piccole (porting 3D)
# ---------------------------------------------------------------------------


def remove_small_components(
    volume: np.ndarray,
    num_classes: int = NUM_CLASSES,
    min_voxels: int = 50,
) -> np.ndarray:
    """Rimuove componenti connesse 3D isolate sotto una soglia di voxel.

    Porting diretto di master/src/eval_utils.py::remove_small_components: la
    logica (26-connettivita', per-classe, soglia configurabile) e' identica —
    qui il volume e' gia' nativamente 3D (nessun adattamento di shape
    necessario, a differenza della migrazione di losses/metrics che ha
    richiesto generalizzazione N-D).

    Args:
        volume:      np.ndarray [H, W, D] etichette intere {0, ..., num_classes-1}.
        num_classes: Numero di classi (per iterare classe per classe).
        min_voxels:  Soglia minima di voxel per mantenere una componente
                     (default 50, stesso valore del vecchio progetto).

    Returns:
        Volume di etichette post-processato, stessa shape/dtype dell'input.
    """
    out = volume.copy()
    struct = scipy.ndimage.generate_binary_structure(3, 3)  # 26-connettivita'

    for cls in range(1, num_classes):  # skip background
        binary_mask = volume == cls
        if not binary_mask.any():
            continue

        labeled_array, n_components = scipy.ndimage.label(binary_mask, structure=struct)
        for comp_id in range(1, n_components + 1):
            component_mask = labeled_array == comp_id
            if int(component_mask.sum()) < min_voxels:
                out[component_mask] = 0

    return out


# ---------------------------------------------------------------------------
# Inferenza 3D nativa per soggetto
# ---------------------------------------------------------------------------


def get_subject_ids(data_dir: str) -> List[str]:
    """Ritorna gli identificatori dei soggetti presenti in una cartella di split.

    Args:
        data_dir: Cartella dello split (es. data/processed_3d/test), con una
                  sottocartella per soggetto.

    Returns:
        Lista ordinata di subject_id.
    """
    return sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )


@torch.no_grad()
def predict_volume_3d(
    model: nn.Module,
    data_dir: str,
    subject_id: str,
    device: torch.device,
    roi: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    apply_postprocessing: bool = True,
    min_component_voxels: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inferenza 3D nativa per un soggetto — nessun patch sampling, nessun crop.

    Args:
        model:       Modello 3D in eval mode, su `device`.
        data_dir:    Cartella dello split (es. data/processed_3d/test).
        subject_id:  Identificatore del soggetto.
        device:      Device CUDA.
        roi:         Dimensione della finestra scorrevole (stessa del training).
        num_classes: Numero di classi.
        sw_batch_size: Finestre processate in parallelo dalla sliding window.
        overlap:     Sovrapposizione tra finestre adiacenti.
        apply_postprocessing: Se True, applica remove_small_components alla
                     predizione prima di ritornarla.
        min_component_voxels: Soglia per remove_small_components.

    Returns:
        pred_vol: np.ndarray [H, W, D] int16 — etichette predette (post-processate
                  se apply_postprocessing=True).
        pred_vol_raw: np.ndarray [H, W, D] int16 — etichette predette GREZZE
                  (pre-post-processing), utile per confrontare l'effetto del
                  post-processing.
        gt_vol:   np.ndarray [H, W, D] int16 — etichette ground-truth (nessun
                  crop: il volume intero, coerentemente con l'inferenza nativa).
    """
    subj_dir = os.path.join(data_dir, subject_id)
    image_paths = [os.path.join(subj_dir, f"{subject_id}-{mod}.nii.gz") for mod in MODALITIES]
    label_path = os.path.join(subj_dir, f"{subject_id}-seg.nii.gz")

    transform = build_eval_transforms(num_classes=num_classes, with_dtm=False)
    data = transform({"image": image_paths, "label": label_path})

    image = data["image"].unsqueeze(0).to(device)  # [1, 4, H, W, D]
    gt = data["label"].squeeze(0).squeeze(0).cpu().numpy().astype(np.int16)  # [H, W, D]

    model.eval()
    logits = sliding_window_inference(
        inputs=image, roi_size=roi, sw_batch_size=sw_batch_size,
        predictor=model, overlap=overlap, mode="gaussian",
    )
    pred_raw = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)  # [H, W, D]

    pred = pred_raw
    if apply_postprocessing:
        pred = remove_small_components(pred_raw, num_classes=num_classes, min_voxels=min_component_voxels)

    return pred, pred_raw, gt


@torch.no_grad()
def predict_probs_3d(
    model: nn.Module,
    data_dir: str,
    subject_id: str,
    device: torch.device,
    roi: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Inferenza 3D nativa per un soggetto, senza argmax: ritorna le probabilita'
    softmax grezze invece delle etichette di classe.

    Variante di predict_volume_3d pensata per l'ENSEMBLE (src/ensemble_3d.py,
    roadmap Sezione 2 §2.1/§2.4): l'argmax interno di predict_volume_3d butta
    via la confidenza relativa del modello, impedendo di mediare piu' modelli
    a livello di probabilita' (soft voting) — qui l'argmax e il post-processing
    restano responsabilita' del CHIAMANTE, dopo l'eventuale fusione tra piu'
    modelli. predict_volume_3d resta INVARIATA (nessuna rottura per i chiamanti
    single-model esistenti, es. evaluate_test_set_3d): questa e' un'aggiunta,
    non un refactor.

    Args:
        model:       Modello 3D in eval mode, su `device`.
        data_dir:    Cartella dello split (es. data/processed_3d/test).
        subject_id:  Identificatore del soggetto.
        device:      Device CUDA.
        roi:         Dimensione della finestra scorrevole (stessa del training).
        num_classes: Numero di classi.
        sw_batch_size: Finestre processate in parallelo dalla sliding window.
        overlap:     Sovrapposizione tra finestre adiacenti.

    Returns:
        probs:  Float tensor [C, H, W, D] su CPU — probabilita' softmax
                (canale 0 = background), NESSUN argmax/post-processing applicato.
        gt_vol: np.ndarray [H, W, D] int16 — etichette ground-truth (volume
                intero, nessun crop).
    """
    subj_dir = os.path.join(data_dir, subject_id)
    image_paths = [os.path.join(subj_dir, f"{subject_id}-{mod}.nii.gz") for mod in MODALITIES]
    label_path = os.path.join(subj_dir, f"{subject_id}-seg.nii.gz")

    transform = build_eval_transforms(num_classes=num_classes, with_dtm=False)
    data = transform({"image": image_paths, "label": label_path})

    image = data["image"].unsqueeze(0).to(device)  # [1, 4, H, W, D]
    gt = data["label"].squeeze(0).squeeze(0).cpu().numpy().astype(np.int16)  # [H, W, D]

    model.eval()
    logits = sliding_window_inference(
        inputs=image, roi_size=roi, sw_batch_size=sw_batch_size,
        predictor=model, overlap=overlap, mode="gaussian",
    )
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()  # [C, H, W, D]

    return probs, gt


# ---------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------


def load_subject_affine(raw_data_dir: str, subject_id: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
    """Carica affine/header dal NIfTI di segmentazione originale del soggetto.

    Args:
        raw_data_dir: Cartella con i NIfTI grezzi del soggetto (stessa
                      struttura di data/processed_3d/<split>/<subject_id>/).
        subject_id:   Identificatore del soggetto.

    Returns:
        (affine, header) del file di segmentazione originale.
    """
    seg_path = os.path.join(raw_data_dir, subject_id, f"{subject_id}-seg.nii.gz")
    img = nib.load(seg_path)
    return img.affine, img.header


def save_prediction_nifti(
    volume: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str,
) -> None:
    """Salva un volume di etichette come NIfTI valido, con l'affine originale.

    Args:
        volume:   np.ndarray [H, W, D] etichette intere.
        affine:   Affine 4x4 del soggetto originale (preserva orientamento/spacing).
        header:   Header NIfTI originale (metadati addizionali).
        out_path: Path di destinazione (.nii.gz).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = nib.Nifti1Image(volume.astype(np.int16), affine, header)
    nib.save(img, out_path)


# ---------------------------------------------------------------------------
# Loop di valutazione completo sul test set
# ---------------------------------------------------------------------------


def evaluate_test_set_3d(
    model: nn.Module,
    data_dir: str,
    device: torch.device,
    roi: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    apply_postprocessing: bool = True,
    min_component_voxels: int = 50,
    export_nifti_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Valuta un modello sull'intero test set con inferenza 3D nativa.

    Args:
        model:       Modello 3D in eval mode, su `device`.
        data_dir:    Cartella dello split test (es. data/processed_3d/test).
        device:      Device CUDA.
        roi:         Dimensione della finestra scorrevole.
        num_classes: Numero di classi.
        apply_postprocessing: Applica remove_small_components alle predizioni.
        min_component_voxels: Soglia per remove_small_components.
        export_nifti_dir: Se fornito, esporta la predizione POST-PROCESSATA
                     di OGNI soggetto come NIfTI in questa cartella (uno per
                     soggetto: "<subject_id>_pred.nii.gz").
        raw_data_dir: Richiesto se export_nifti_dir e' fornito — cartella con
                     i NIfTI grezzi originali, per recuperare affine/header
                     corretti (tipicamente la stessa data_dir, dato che
                     scripts/reorganize_volumes.py copia i file grezzi senza
                     modificarli).

    Returns:
        Dict {subject_id: {dice_ET: ..., hd95_ET: ..., ...}} — metriche
        per-soggetto, per-classe (nessuna aggregazione qui: l'aggregazione
        statistica sul test set e' responsabilita' del chiamante/notebook di
        confronto).
    """
    if export_nifti_dir is not None and raw_data_dir is None:
        raise ValueError("raw_data_dir e' richiesto quando export_nifti_dir e' fornito.")

    subject_ids = get_subject_ids(data_dir)
    print(f"Valutazione 3D nativa su {len(subject_ids)} soggetti test...")

    per_subject_metrics: Dict[str, Dict[str, float]] = {}

    for i, subject_id in enumerate(subject_ids):
        pred, pred_raw, gt = predict_volume_3d(
            model, data_dir, subject_id, device, roi=roi, num_classes=num_classes,
            apply_postprocessing=apply_postprocessing, min_component_voxels=min_component_voxels,
        )

        metrics = SegmentationMetrics3D(num_classes=num_classes)
        pred_t = torch.from_numpy(pred).long().unsqueeze(0)
        gt_t = torch.from_numpy(gt).long().unsqueeze(0)
        # SegmentationMetrics3D.update si aspetta logits [B,C,*spatial]: qui
        # la predizione e' gia' un argmax, quindi la trasformiamo in
        # "pseudo-logits" one-hot scalati (equivalenti ad argmax dopo softmax).
        pseudo_logits = torch.nn.functional.one_hot(pred_t, num_classes=num_classes).permute(
            0, 4, 1, 2, 3
        ).float() * 10.0
        metrics.update(pseudo_logits, gt_t)
        per_subject_metrics[subject_id] = metrics.aggregate_and_reset()

        if export_nifti_dir is not None:
            affine, header = load_subject_affine(raw_data_dir, subject_id)
            out_path = os.path.join(export_nifti_dir, f"{subject_id}_pred.nii.gz")
            save_prediction_nifti(pred, affine, header, out_path)

        print(f"  [{i+1}/{len(subject_ids)}] {subject_id}  "
              f"dice_fg={per_subject_metrics[subject_id]['dice_mean_fg']:.4f}", end="\r")

    print()
    return per_subject_metrics


def summarize_metrics(per_subject_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggrega le metriche per-soggetto in medie/std sul test set.

    Args:
        per_subject_metrics: Output di evaluate_test_set_3d.

    Returns:
        Dict con "<metric>_mean" e "<metric>_std" per ciascuna metrica
        presente (dice_ET, hd95_ET, ..., dice_mean_fg, hd95_mean_fg). Le
        medie di HD95 ignorano i NaN (classe assente in quel soggetto).
    """
    if not per_subject_metrics:
        return {}

    keys = next(iter(per_subject_metrics.values())).keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in per_subject_metrics.values()], dtype=np.float64)
        finite = values[~np.isnan(values)]
        summary[f"{key}_mean"] = float(np.mean(finite)) if len(finite) > 0 else float("nan")
        summary[f"{key}_std"] = float(np.std(finite)) if len(finite) > 0 else float("nan")
    return summary
