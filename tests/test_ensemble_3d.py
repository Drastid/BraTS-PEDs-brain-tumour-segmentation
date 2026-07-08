"""
tests/test_ensemble_3d.py
============================
Suite pytest per src/ensemble_3d.py (roadmap Sezione 2, item C4: "test unitari
dell'ensemble su tensori sintetici").

Copre SOLO la logica di fusione (average_probabilities) e il confronto
statistico (compare_ensemble_vs_best_single / ensemble_acceptance_verdict) —
tutta la parte che NON richiede caricare modelli/pesi/volumi reali. La
pipeline di inferenza end-to-end (predict_volume_3d_ensemble,
evaluate_test_set_3d_ensemble) e' gia' coperta indirettamente: riusa
predict_probs_3d/remove_small_components/SegmentationMetrics3D, gia' testati
altrove (tests/test_eval_3d.py, tests/test_metrics_3d.py), e va verificata
end-to-end con dati reali (fuori dallo scope di uno unit test sintetico).

Casi verificati (esplicitamente richiesti dalla roadmap):
    1. Media di 2 softmax identiche = identita'.
    2. Predizioni concordi tra i modelli -> stessa label dopo fusione+argmax,
       indipendentemente dalla confidenza relativa.
    3. Perimetro di shape preservato dalla fusione.
    Piu': media pesata (Opzione B, gia' supportata dall'API anche se non
    ancora usata di default), casi limite (lista vuota, pesi di lunghezza
    sbagliata).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ensemble_3d import (  # noqa: E402
    average_probabilities,
    compare_ensemble_vs_best_single,
    ensemble_acceptance_verdict,
)

NUM_CLASSES = 5


def _random_probs(shape: tuple[int, ...]) -> torch.Tensor:
    """Genera un tensore [C, *spatial] che e' una distribuzione di probabilita'
    valida lungo il canale 0 (softmax di logit casuali)."""
    logits = torch.randn(shape)
    return torch.softmax(logits, dim=0)


# ---------------------------------------------------------------------------
# 1. Media di probabilita' identiche = identita'
# ---------------------------------------------------------------------------


def test_average_probabilities_identical_inputs_is_identity() -> None:
    probs = _random_probs((NUM_CLASSES, 8, 8, 8))
    fused = average_probabilities([probs, probs])
    assert torch.allclose(fused, probs, atol=1e-6)


def test_average_probabilities_three_identical_inputs_is_identity() -> None:
    """Generalizza a N>2 modelli (l'API non e' vincolata a esattamente 2)."""
    probs = _random_probs((NUM_CLASSES, 6, 6, 6))
    fused = average_probabilities([probs, probs, probs])
    assert torch.allclose(fused, probs, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Predizioni concordi -> stessa label dopo fusione+argmax
# ---------------------------------------------------------------------------


def test_concordant_predictions_survive_fusion() -> None:
    """Se i due modelli concordano sulla classe argmax in ogni voxel (con
    confidenza DIVERSA), la fusione+argmax deve produrre la stessa mappa di
    label — la media di probabilita' non deve "confondere" un accordo netto.
    """
    spatial = (2, 5, 5, 5)  # [C, D, H, W]
    target_labels = torch.randint(0, NUM_CLASSES, spatial[1:])

    def _confident_probs(sharpness: float) -> torch.Tensor:
        # One-hot "morbido": la classe target ha probabilita' alta, le altre
        # si spartiscono il resto — concordanza sull'argmax garantita per
        # costruzione, ma con confidenza (sharpness) diversa tra i due "modelli".
        one_hot = torch.nn.functional.one_hot(target_labels, num_classes=NUM_CLASSES)
        one_hot = one_hot.permute(3, 0, 1, 2).float()  # [C, D, H, W]
        uniform = torch.full_like(one_hot, 1.0 / NUM_CLASSES)
        return sharpness * one_hot + (1 - sharpness) * uniform

    probs_a = _confident_probs(sharpness=0.9)   # modello molto sicuro
    probs_b = _confident_probs(sharpness=0.35)  # modello poco sicuro, ma concorde

    fused = average_probabilities([probs_a, probs_b])
    pred = fused.argmax(dim=0)

    assert torch.equal(pred, target_labels)


def test_disagreement_resolved_by_higher_confidence() -> None:
    """Se i due modelli sono in disaccordo netto su un voxel, la fusione deve
    seguire il modello PIU' sicuro (media pesata implicita dalla confidenza
    stessa — proprieta' base del soft voting)."""
    # 3 classi per semplicita' di lettura: [C]
    probs_a = torch.tensor([0.05, 0.05, 0.90])  # modello A: sicurissimo su classe 2
    probs_b = torch.tensor([0.10, 0.80, 0.10])  # modello B: abbastanza sicuro su classe 1, ma meno di A

    fused = average_probabilities([probs_a, probs_b])
    # media: [0.075, 0.425, 0.500] -> argmax = classe 2 (quella su cui A era piu' sicuro)
    assert fused.argmax(dim=0).item() == 2


# ---------------------------------------------------------------------------
# 3. Perimetro di shape preservato
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 4, 4, 4), (5, 16, 16, 16), (3, 10, 12, 8)])
def test_average_probabilities_preserves_shape(shape: tuple[int, ...]) -> None:
    probs_a = _random_probs(shape)
    probs_b = _random_probs(shape)
    fused = average_probabilities([probs_a, probs_b])
    assert tuple(fused.shape) == shape


def test_average_probabilities_output_sums_to_one_per_voxel() -> None:
    """La media di distribuzioni di probabilita' valide e' ancora una
    distribuzione di probabilita' valida (somma 1 lungo il canale)."""
    probs_a = _random_probs((NUM_CLASSES, 4, 4, 4))
    probs_b = _random_probs((NUM_CLASSES, 4, 4, 4))
    fused = average_probabilities([probs_a, probs_b])
    sums = fused.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Media pesata (DEC-1 Opzione B, gia' supportata dall'API)
# ---------------------------------------------------------------------------


def test_average_probabilities_weighted_matches_manual_computation() -> None:
    probs_a = torch.tensor([0.2, 0.8])
    probs_b = torch.tensor([0.6, 0.4])
    # pesi non normalizzati (3:1) -> normalizzati internamente a (0.75, 0.25)
    fused = average_probabilities([probs_a, probs_b], weights=[3.0, 1.0])
    expected = 0.75 * probs_a + 0.25 * probs_b
    assert torch.allclose(fused, expected, atol=1e-6)


def test_average_probabilities_weights_wrong_length_raises() -> None:
    probs_a = _random_probs((NUM_CLASSES, 4, 4, 4))
    probs_b = _random_probs((NUM_CLASSES, 4, 4, 4))
    with pytest.raises(ValueError):
        average_probabilities([probs_a, probs_b], weights=[1.0])


def test_average_probabilities_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        average_probabilities([])


# ---------------------------------------------------------------------------
# Confronto statistico (DEC-4 Wilcoxon) e verdetto (DEC-5)
# ---------------------------------------------------------------------------


def _fake_per_subject(dice_values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {subj: {"dice_mean_fg": v} for subj, v in dice_values.items()}


def test_compare_ensemble_vs_best_single_detects_improvement() -> None:
    ensemble = _fake_per_subject({"s1": 0.70, "s2": 0.72, "s3": 0.68, "s4": 0.75, "s5": 0.71,
                                   "s6": 0.69, "s7": 0.73, "s8": 0.74})
    single = _fake_per_subject({"s1": 0.60, "s2": 0.62, "s3": 0.58, "s4": 0.65, "s5": 0.61,
                                 "s6": 0.59, "s7": 0.63, "s8": 0.64})

    comparison = compare_ensemble_vs_best_single(ensemble, single, metric_key="dice_mean_fg")
    assert comparison["n_subjects"] == 8
    assert comparison["mean_diff"] > 0

    verdict = ensemble_acceptance_verdict(comparison, alpha=0.05)
    assert verdict["improved"] is True
    assert verdict["accepted"] is True  # miglioramento netto e consistente su 8/8 soggetti


def test_compare_ensemble_vs_best_single_no_common_subjects_raises() -> None:
    ensemble = _fake_per_subject({"s1": 0.7})
    single = _fake_per_subject({"s2": 0.6})
    with pytest.raises(ValueError):
        compare_ensemble_vs_best_single(ensemble, single)


def test_compare_ensemble_vs_best_single_identical_values_not_significant() -> None:
    """Se ensemble e singolo hanno esattamente gli stessi valori (nessuna
    differenza), il test non deve sollevare un'eccezione (scipy.stats.wilcoxon
    lo farebbe su differenze tutte zero) e il verdetto deve essere "non
    accettato" (nessun miglioramento)."""
    values = {"s1": 0.7, "s2": 0.6, "s3": 0.65}
    ensemble = _fake_per_subject(values)
    single = _fake_per_subject(values)

    comparison = compare_ensemble_vs_best_single(ensemble, single)
    assert comparison["mean_diff"] == pytest.approx(0.0)

    verdict = ensemble_acceptance_verdict(comparison)
    assert verdict["accepted"] is False


def test_compare_ensemble_vs_best_single_ignores_nan_subjects() -> None:
    """Un soggetto con dice_mean_fg NaN in uno dei due lati (es. classe
    foreground assente in quel soggetto) va escluso dal confronto, non
    propagato come NaN nella media."""
    ensemble = _fake_per_subject({"s1": 0.7, "s2": float("nan"), "s3": 0.75})
    single = _fake_per_subject({"s1": 0.6, "s2": 0.5, "s3": 0.65})

    comparison = compare_ensemble_vs_best_single(ensemble, single)
    assert comparison["n_subjects"] == 2  # s2 escluso (NaN lato ensemble)
    assert not (comparison["mean_ensemble"] != comparison["mean_ensemble"])  # non-NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
