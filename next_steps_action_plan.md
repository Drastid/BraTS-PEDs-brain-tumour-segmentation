# Next Steps Action Plan — BraTS-PEDs Brain Tumour Segmentation

**Author:** `/code-reviewer` skill (refresh)
**Date:** 2026-05-15
**Companion document:** `current_progress_review.md`
**Supersedes:** 2026-05-11 action plan (TASK 1–7, 10 remain ✅; TASK 8, 9 still as below; TASK 11–14 are new)

The project is **functionally complete and shippable**. This plan tracks the residual work to take it from ~99% to 100% of the original spec, plus optional polish. **No item below is a correctness bug.** TASK 11 is the only newly identified substantive item; it improves the *honesty* of the headline numbers without changing them. TASK 9 remains the only true blocker, gated on external data.

---

## Roadmap Overview — current state

| # | Track | Status | Commit / Note |
|---|---|---|---|
| 1 | Implement missing IoU metric | ✅ done | `78070bf` |
| 2 | Move `import warnings` to module-level | ✅ done | `e4eb77b` |
| 3 | Centralise constants in `src/constants.py` | ✅ done | `6a87ab2` |
| 4 | Add unit tests for losses, eval_utils, dataset (9 tests) | ✅ done | `397a0a8` |
| 5 | Re-run 3D test eval with IoU after Task 1 (CUDA) | ✅ done | `1dd54e3` |
| 6 | Save NIfTI predictions for clinical viewing (6 files) | ✅ done | `033d06f` |
| 7 | Reconcile output directories → `training_outputs/` | ✅ done | `d3f943d` |
| 8 | Delete unused venvs `.venv-1`, `.venv-2` | 🟡 still pending user confirmation | — |
| 9 | Cross-domain zero-shot evaluation | 🔴 blocker (data) | — |
| 10 | Final report polish (IoU + 2D-vs-3D clarification) | ✅ done | `3d00c57` |
| **11** | **Surface Dice/IoU degenerate-case counts** | 🟡 **NEW** — see below | — |
| **12** | **Resolve `06b_evaluation_fpn.ipynb` stub** | 🟢 **NEW** — see below | — |
| **13** | Pin `requirements.txt` versions | 🟢 optional | — |
| **14** | Expand test coverage (4 cheap additions) | 🟢 optional | — |

**Effort to ship at 100% (spec-required):** zero — TASK 9 is gated by external data; TASK 11 improves reporting honesty but is not a spec requirement.

---

## TASK 11 — Surface Dice/IoU degenerate-case counts 🟡 (new)

**Why.** `compute_dice_volume` and `compute_iou_volume` return **1.0** when a class is absent in both prediction and ground truth (Laplace smoothing `ε = 1e-6` makes `(0+ε)/(0+ε) = 1.0`). The per-class mean in `evaluation_outputs/test_3d_metrics_*.json` and `README.md` silently includes these inflating `1.0`s. HD95 already handles this correctly (NaN + `np.nanmean` + `hd95_nan_count` reported per class). Dice/IoU should do the same.

Empirics on the current test set (subjects with `dice = 1.0` ∧ `hd95 = None`):

| Model | NCR | ET | (ED) |
|---|---|---|---|
| U-Net | 6 / 26 | 11 / 26 | 0 |
| FPN | 7 / 26 | 13 / 26 | 0 |
| SegFormer | 7 / 26 | 14 / 26 | 0 |

Effect: per-class ET Dice means drop from ~0.54–0.63 (reported) to ~0.06–0.20 when re-computed on subjects where ET actually appears. Model ranking is unchanged — every model is hit the same way.

**Plan.**

1. **`src/evaluate_3d_test.py` — `evaluate_3d_on_test`**: after each subject loop iteration, count `dice_{cls} == 1.0 and hd95_{cls} is np.nan` (Case A). Add `dice_degenerate_count` and `iou_degenerate_count` to the per-class summary rows (mirror the existing `HD95 NaN count` column).
2. **`src/evaluate_3d_test.py` — `save_results`**: include `dice_degenerate_count` and `iou_degenerate_count` in `summary[cls]` of the JSON.
3. **Optional (recommended)**: add a `dice_mean_present` and `iou_mean_present` to `summary[cls]` — mean over subjects where the class is present in either pred or GT. Two new keys per class; small, useful, no recomputation cost.
4. **`README.md`**: under the 3D Volumetric Metrics table, add a note like *"Dice and IoU per class are means over all 26 subjects; subjects where the class is absent in both prediction and ground truth contribute a smoothing-induced 1.0. The number of such subjects (`*_degenerate_count`) is recorded in the per-model JSON. The model ranking is invariant to this choice."* Optionally surface a second row in the table for `Dice (present-class only)` to give the more conservative read.
5. **Tests**: add 2 tests to `tests/test_eval_utils.py`:
   - `test_dice_degenerate_both_empty()`: assert `compute_dice_volume(zeros, zeros)["dice_NCR"] == pytest.approx(1.0)` — locks in the convention.
   - `test_iou_degenerate_both_empty()`: same for IoU.

**Acceptance.** The three JSONs in `evaluation_outputs/` carry `dice_degenerate_count` and `iou_degenerate_count` (and optionally `*_mean_present`); `README.md` explains the convention; 11 tests passing.

**Estimated effort.** 30–45 minutes including re-running `python -m src.evaluate_3d_test --model all` on `.venv-3` to regenerate the JSONs. No retraining required.

**Important.** Do **not** change the existing `compute_dice_volume` / `compute_iou_volume` return value — keep `1.0` for the degenerate case to preserve backward compatibility with `08_comparison.ipynb` and any external consumers of the JSONs. Add the counts alongside the existing means; do not replace them.

---

## TASK 9 — Cross-domain zero-shot evaluation 🔴 (data-dependent — unchanged from prior plan)

**Why.** `notebooks/07_cross_domain_evaluation.ipynb` is fully written but never executed. Running it on an adult BraTS dataset would quantify the paediatric → adult domain shift, strengthening any final report.

**Blocker.** Requires an adult BraTS dataset (e.g. BraTS2023-GLI). Not currently in the working directory.

**Plan once data is available.**

1. Place the adult NIfTI files under `PKG - BraTS-2023-GLI/` (or similar).
2. Re-run `notebooks/02_preprocessing.ipynb` with `DATA_ROOT` and `OUTPUT_ROOT` pointed at the new dataset. Output goes to `processed_dataset_adult/`. (~20 min, CPU-bound — fine in `.venv`.)
3. In `notebooks/07_cross_domain_evaluation.ipynb`, set `NEW_DATASET_PATH = "processed_dataset_adult/test"`. The notebook already handles different spatial dims via `infer_dataset_shape()`.
4. Run all cells in `.venv-3` (CUDA). Saves comparison plots + per-subject CSV under `training_outputs/`.
5. Add a one-paragraph cross-domain section to `README.md` summarising the in-domain → cross-domain Dice/HD95 gap per model.

**Acceptance.** Notebook fully executed; per-model in-domain vs cross-domain delta documented.

**Estimated effort.** Half-day, mostly preprocessing + reading output.

---

## TASK 12 — Resolve `06b_evaluation_fpn.ipynb` stub 🟢 (new, cleanup)

**Why.** The notebook has **zero executed code cells** today. Its content is a thin wrapper that calls `python -m src.evaluate_3d_test --model fpn` via `subprocess.run` and then loads the resulting JSON for display. Since `evaluate_3d_test.py` is the canonical entry point (and is what `README.md` documents first), the notebook adds nothing.

**Options (pick one).**

- **Option A — Execute and commit.** Run all cells in `.venv-3`, save outputs into the notebook, commit. Pros: matches the pattern of the other executed notebooks; gives a self-contained record of the FPN evaluation. Cons: 1 more committed notebook with embedded outputs.
- **Option B — Delete.** Remove the file; `README.md` (which currently lists `notebooks/06b_evaluation_fpn.ipynb` in the Project Structure tree) loses one entry. Pros: simpler. Cons: loses the convenience wrapper.

**Recommendation.** Option B (delete + update README). The CLI script is the source of truth, and `08_comparison.ipynb` already reads `evaluation_outputs/test_3d_metrics_fpn.json` for display purposes.

**Acceptance.** Either the notebook is executed with non-empty outputs, or the file is removed and the README tree updated.

**Estimated effort.** 5 minutes (option B); 10 minutes (option A).

---

## TASK 8 — Delete unused virtual environments 🟡 (still pending user confirmation)

**Status.** Same as 2026-05-11 plan.

| Venv | Status | Action |
|---|---|---|
| `.venv` | CPU-only (`torch 2.11.0+cpu`) | Keep — fast for non-GPU work. Optionally `pip install pytest` if you want to run the test suite there too. |
| `.venv-1` | CUDA torch but missing `transformers`/`accelerate` | **Safe to delete** |
| `.venv-2` | Same as `.venv-1` | **Safe to delete** |
| `.venv-3` | Full CUDA + transformers + accelerate + pytest | **Keep — primary GPU env** |
| `.venv-4` | Full CUDA + transformers + accelerate | **Keep — backup GPU env** |

**Implementation (user confirmation required).**
```powershell
Remove-Item -Recurse -Force .venv-1, .venv-2
```

**Acceptance.** `.venv`, `.venv-3`, `.venv-4` remain; `python -c "import torch; print(torch.cuda.is_available())"` returns `True` inside `.venv-3`.

**Estimated effort.** 1 minute.

---

## TASK 13 — Pin `requirements.txt` versions 🟢 (optional)

**Why.** `requirements.txt` currently lists package names with no version constraints. `transformers` 5.x (used here) is a major-version release; future installs against a clean interpreter may break `src/models.py` (`SegformerForSemanticSegmentation` import surface changed historically). Cheap insurance to lock the working set.

**Plan.**

```bash
# Inside .venv-3 (primary GPU env)
.venv-3\Scripts\activate
pip freeze > requirements-lock.txt
# Then hand-edit requirements.txt to set ==<version> for the top-level deps,
# OR commit requirements-lock.txt as the authoritative pinned manifest
# and keep requirements.txt as the loose top-level dependency list.
```

**Acceptance.** A reproducible install of the primary GPU env via `pip install -r requirements-lock.txt`. Bonus: a single line in `README.md` Setup pointing users at the lock file for reproducibility.

**Estimated effort.** 5 minutes.

---

## TASK 14 — Expand test coverage 🟢 (optional)

**Why.** The 9 existing tests cover the most error-prone primitives, but a handful of cheap additions would close meaningful gaps without slowing CI.

**Suggested additions (≤ 5 minutes each, all CPU-friendly).**

1. **`tests/test_eval_utils.py::test_dice_degenerate_both_empty`** — assert the documented `1.0`-for-both-empty behaviour. (Pairs with TASK 11.)
2. **`tests/test_eval_utils.py::test_iou_degenerate_both_empty`** — same for IoU.
3. **`tests/test_models.py::test_segformer_4ch_patch_embedding`** — call `get_segformer(num_classes=4)`, assert `model.encoder.patch_embeddings[0].proj.weight.shape[1] == 4` and `weight[:, 3] == weight[:, :3].mean(dim=1)`. Catches future regressions in the 4-channel adaptation.
4. **`tests/test_train_utils.py::test_center_crop_offsets`** — assert `center_crop(zeros(1,4,240,240), zeros(1,240,240), 192)` returns shapes `(1,4,192,192)` and `(1,192,192)`, and that the top-left of the cropped tensor maps to original index `(24, 24)`. Locks in Critical Invariant #2.
5. **`tests/test_train_utils.py::test_class_weights_sum_to_one`** — assert `get_class_weights().sum().item() == pytest.approx(1.0, abs=1e-6)`.

**Acceptance.** ≥ 13 tests passing in `.venv-3`.

**Estimated effort.** 20 minutes total, less if items 1–2 are done as part of TASK 11.

---

## Lower-priority polish (only if you find yourself bored)

- **Fix `evaluate_3d_test.py` docstring** (line 167): currently says *"U-Net" or "SegFormer"* — should say *"U-Net", "FPN", or "SegFormer"*. Cosmetic.
- **Add `src/export_nifti_predictions.py` and `src/constants.py`** to the Project Structure tree in `README.md`. They are listed nowhere in the tree today, even though both are committed and used.
- **GitHub Actions workflow** that runs `pytest -q` on every push. ~10 lines of YAML; the test suite is fast (14 s).
- **Replace `print(...)` with `logging.info(...)`** in `src/*.py`. Useful if the CLI scripts are ever run from a logger-aware orchestrator. Probably overkill for an academic project.

---

## Out of scope — do not attempt

- **Changing the `compute_dice_volume` / `compute_iou_volume` return value for the both-empty case.** Backward compatibility with `08_comparison.ipynb` and the committed JSONs matters; add reporting around the existing convention instead (TASK 11).
- **Renaming `processing_02_outputs/`** → unnecessary churn across 5 source files.
- **Adding pixel-level augmentation** → violates Critical Invariant #1.
- **Re-running training** to get new checkpoints → no benefit; current results are documented and committed.
- **Deleting `.venv-3` or `.venv-4`** → they are the only working CUDA environments.
- **Reformatting notebook outputs** in `07_cross_domain_evaluation.ipynb` — they will refresh on the next execution.

---

## Suggested order if completing today

1. **TASK 11** (Dice/IoU degenerate-case reporting): 30–45 min. Adds two JSON keys per class, a `README.md` note, and 2 tests. Re-runs `evaluate_3d_test --model all` on `.venv-3` to regenerate the JSONs.
2. **TASK 14 items 1–2** done as part of TASK 11; item 3 (SegFormer 4-ch test) optional.
3. **TASK 12** (delete `06b_evaluation_fpn.ipynb` + update README tree): 5 min.
4. **TASK 8** (delete `.venv-1`, `.venv-2`): user confirms, 1 min.
5. **TASK 13** (pin requirements): 5 min.
6. **STOP unless adult BraTS data arrives.** The project is shippable today.

If/when adult BraTS data arrives, follow TASK 9's plan.

---

## Summary

The project is **complete relative to its original specification** modulo the data-gated TASK 9. The only newly identified substantive item is **TASK 11** — surface the silent Dice/IoU degenerate-case `1.0`s in the same way HD95 already surfaces its NaN count. This is a reporting honesty improvement; it does **not** invalidate any committed result and is **not** a correctness bug.

- **Critical** (must do for spec compliance): none.
- **Strongly recommended**: TASK 11 (reporting honesty), TASK 9 (if adult BraTS data available).
- **Nice-to-have**: TASK 12 (notebook cleanup), TASK 8 (venv cleanup), TASK 13 (pin requirements), TASK 14 (4 cheap tests).
- **Stretch**: GitHub Actions CI; `print → logging`.

After completing TASK 11, TASK 12, and the cosmetic fixes noted under "Lower-priority polish", the project moves from ~99% → **100%** relative to its original specification.
