"""
tests/test_train_3d.py
=========================
Verifica src/train_3d.py::EarlyStopper3D e monitor_value — porting del
pattern master/run_pipeline.py::_EarlyStopper per la pipeline 3D, che prima
non aveva alcuna logica anti-overfitting oltre al weight decay.

Logica pura (nessuna dipendenza da modello/dataset/GPU): eseguibile ovunque.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_3d import EarlyStopper3D, monitor_value


# ---------------------------------------------------------------------------
# EarlyStopper3D
# ---------------------------------------------------------------------------


def test_disabled_never_stops() -> None:
    stopper = EarlyStopper3D(enabled=False, patience=2, min_delta=0.0)
    for epoch, value in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
        assert stopper.update(value, epoch) is False


def test_patience_zero_disables_even_if_enabled_true() -> None:
    """enabled=True ma patience<=0 deve comportarsi come disabilitato (stessa
    semantica di master/run_pipeline.py::_EarlyStopper)."""
    stopper = EarlyStopper3D(enabled=True, patience=0, min_delta=0.0)
    for epoch, value in enumerate([0.5, 0.4, 0.3]):
        assert stopper.update(value, epoch) is False


def test_stops_after_patience_non_improving_epochs() -> None:
    stopper = EarlyStopper3D(enabled=True, patience=3, min_delta=0.0)
    # Epoca 0: primo valore, sempre "miglioramento" (best parte da -inf).
    assert stopper.update(0.50, 0) is False
    # Epoche 1-2: non migliorano -> num_bad sale a 1, poi 2 (< patience=3).
    assert stopper.update(0.40, 1) is False
    assert stopper.update(0.45, 2) is False
    # Epoca 3: terzo non-miglioramento consecutivo -> num_bad=3 >= patience=3.
    assert stopper.update(0.30, 3) is True
    assert stopper.best == 0.50
    assert stopper.best_epoch == 0


def test_improvement_resets_patience_counter() -> None:
    stopper = EarlyStopper3D(enabled=True, patience=2, min_delta=0.0)
    assert stopper.update(0.50, 0) is False
    assert stopper.update(0.40, 1) is False  # num_bad=1
    assert stopper.update(0.60, 2) is False  # miglioramento -> num_bad torna a 0
    assert stopper.update(0.55, 3) is False  # num_bad=1 (< patience=2)
    assert stopper.update(0.50, 4) is True   # num_bad=2 >= patience=2
    assert stopper.best == 0.60
    assert stopper.best_epoch == 2


def test_min_delta_requires_meaningful_improvement() -> None:
    """Un incremento piu' piccolo di min_delta NON conta come miglioramento."""
    stopper = EarlyStopper3D(enabled=True, patience=2, min_delta=0.05)
    assert stopper.update(0.50, 0) is False
    # +0.01 < min_delta=0.05 -> non e' un miglioramento valido, num_bad=1.
    assert stopper.update(0.51, 1) is False
    assert stopper.update(0.52, 2) is True  # num_bad=2 >= patience=2
    assert stopper.best == 0.50


def test_best_epoch_tracks_last_true_improvement() -> None:
    stopper = EarlyStopper3D(enabled=True, patience=10, min_delta=0.0)
    stopper.update(0.10, 0)
    stopper.update(0.30, 1)
    stopper.update(0.20, 2)
    stopper.update(0.30, 3)  # pareggio, non supera best+min_delta -> non aggiorna
    assert stopper.best == 0.30
    assert stopper.best_epoch == 1


# ---------------------------------------------------------------------------
# monitor_value
# ---------------------------------------------------------------------------


def _fake_history(values: list[float]) -> list[dict]:
    return [{"epoch": i, "val": {"dice_mean_fg": v}} for i, v in enumerate(values)]


def test_monitor_value_window_1_returns_last_epoch_raw() -> None:
    history = _fake_history([0.1, 0.5, 0.9])
    assert monitor_value(history, window=1) == 0.9


def test_monitor_value_smooths_over_window() -> None:
    history = _fake_history([0.1, 0.5, 0.9])
    # Media delle ultime 2 epoche: (0.5 + 0.9) / 2 = 0.7
    assert monitor_value(history, window=2) == 0.7


def test_monitor_value_window_larger_than_history_uses_all_available() -> None:
    history = _fake_history([0.2, 0.4])
    assert monitor_value(history, window=10) == (0.2 + 0.4) / 2


def test_monitor_value_window_zero_or_negative_falls_back_to_one() -> None:
    history = _fake_history([0.1, 0.5, 0.9])
    assert monitor_value(history, window=0) == 0.9
    assert monitor_value(history, window=-5) == 0.9


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
