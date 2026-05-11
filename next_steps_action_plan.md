# Next Steps Action Plan — BraTS-PEDs Brain Tumour Segmentation

**Author:** Code-Reviewer Skill (refresh)
**Date:** 2026-05-11
**Companion document:** `current_progress_review.md`
**Previous plan:** same date, earlier in the day (superseded — all 🔴/🟡 items are now done)

This document tracks the only remaining work to bring the project to **100% completion** relative to its original specification, plus optional polish. **Everything 🔴/🟡 from the prior plan is now closed**; what remains is a data-dependent blocker and a small set of optional cleanups.

---

## Roadmap Overview — current state

| # | Track | Status | Commit |
|---|---|---|---|
| 1 | Implement missing IoU metric | ✅ done | `78070bf` |
| 2 | Move `import warnings` to module-level | ✅ done | `e4eb77b` |
| 3 | Centralise constants in `src/constants.py` | ✅ done | `6a87ab2` |
| 4 | Add unit tests for losses, eval_utils, dataset (9 tests) | ✅ done | `397a0a8` |
| 5 | Re-run 3D test eval with IoU after Task 1 (CUDA) | ✅ done | `1dd54e3` |
| 6 | Save NIfTI predictions for clinical viewing (6 files) | ✅ done | `033d06f` |
| 7 | Reconcile output directories → `training_outputs/` | ✅ done | `d3f943d` |
| 8 | Delete unused venvs | 🟡 **partial — see TASK 8 below** | — |
| 9 | Cross-domain zero-shot evaluation | 🔴 **blocker (data)** | — |
| 10 | Final report polish (IoU + 2D-vs-3D clarification) | ✅ done | `3d00c57` |

**Effort to reach 100%:** zero — except TASK 9, which is gated by external data.

---

## TASK 9 — Cross-domain zero-shot evaluation 🔴 (data-dependent)

**Why:** `07_cross_domain_evaluation.ipynb` is fully written but has never been executed. Running it on an adult BraTS dataset would quantify the paediatric → adult domain shift — a meaningful generalisation experiment that strengthens any final report.

**Blocker:** requires an adult BraTS dataset (e.g. BraTS2023-GLI, BraTS2024). Not currently in the working directory.

**Plan once data is available:**

1. Place the adult NIfTI files under `PKG - BraTS-2023-GLI/` (or similar).
2. Re-run `02_preprocessing.ipynb` with `DATA_ROOT` and `OUTPUT_ROOT` pointed at the new dataset. Output goes to `processed_dataset_adult/`. (~20 min, CPU-bound — fine in `.venv`.)
3. In `07_cross_domain_evaluation.ipynb`, set `NEW_DATASET_PATH = "processed_dataset_adult/test"`. The notebook is already built to handle different spatial dims via `infer_dataset_shape()`.
4. Run all cells in `.venv-3` (CUDA). Saves comparison plots + per-subject CSV under `training_outputs/`.
5. Add a one-paragraph cross-domain section to `README.md` summarising the in-domain → cross-domain Dice/HD95 gap per model.

**Acceptance:** `07_cross_domain_evaluation.ipynb` is fully executed; per-model in-domain vs cross-domain delta documented.

**Estimated effort:** half-day (mostly preprocessing + reading output).

---

## TASK 8 — Delete unused virtual environments 🟡 (corrected from prior plan)

**Why:** Disk hygiene. Each `.venv-*` directory is several GB.

**Correction from the previous action plan:** the prior plan claimed all four (`.venv-1`–`.venv-4`) were dead. **That is false.**

| Venv | Status | Action |
|---|---|---|
| `.venv` | CPU-only (`torch 2.11.0+cpu`) | Keep — fast for non-GPU work |
| `.venv-1` | CUDA torch but missing `transformers`/`accelerate` | **Safe to delete** |
| `.venv-2` | Same as `.venv-1` | **Safe to delete** |
| `.venv-3` | Full CUDA + transformers + accelerate + pytest | **Keep — primary GPU env** |
| `.venv-4` | Full CUDA + transformers + accelerate | **Keep — backup GPU env** |

**Implementation (user-confirmation required before deletion):**
```powershell
Remove-Item -Recurse -Force .venv-1, .venv-2
```

**Acceptance:** `.venv`, `.venv-3`, `.venv-4` remain; `python -c "import torch; print(torch.cuda.is_available())"` returns `True` inside `.venv-3`.

**Estimated effort:** 1 minute (user confirms; deletion is instant).

---

## Optional polish (not blocking shipping)

### Update `CLAUDE.md` §8 "Known Open Items"
**Current text:** lists IoU, `import warnings`, constants centralisation as the "highest-priority remaining". All three are now closed (commits `78070bf`, `e4eb77b`, `6a87ab2`).
**Suggested replacement:** "Highest-priority remaining work: TASK 9 cross-domain evaluation (blocked on adult BraTS data). All code-side spec gaps are closed as of 2026-05-11."

(This file will be updated in the same commit as this action plan.)

### Annotate `project_review.md` as superseded
**Current state:** dated 2026-04-29; all 🔴 items it flagged have been resolved.
**Suggested action:** add a one-line "**Status:** superseded by `current_progress_review.md` (2026-05-11) — every 🔴/🟡 item below is closed" at the top so future readers know it is historical.

### Optional: integration test that actually runs `evaluate_3d_test --model segformer`
**Why:** the existing 9 unit tests cover individual functions but not the full pipeline. A single integration test that loads `checkpoints/segformer/best.pth` and runs prediction on one test subject would catch any future regression in the predict → post-process → JSON-save chain.
**Why probably skip:** requires `.pth` files (git-ignored), takes ~30 s per test subject, and the existing `python -m src.evaluate_3d_test --model segformer` already serves as a manual smoke test.

### Optional: `tests/test_constants.py`
**Why:** trivial sanity test (`assert NUM_CLASSES == len(CLASS_NAMES)`, `assert VOXEL_FREQ.sum() ≈ 1.0`). 5 lines. Defends against future edits.

---

## Out of scope (do not attempt)

- **Renaming `processing_02_outputs/`** → unnecessary churn across 5 source files.
- **Adding pixel-level augmentation** → violates Critical Invariant #1.
- **Re-running training** to get new checkpoints → no benefit; current results are documented and committed.
- **Deleting `.venv-3` or `.venv-4`** → they are the ONLY working CUDA environments.
- **Reformatting notebook outputs** (the stale cell outputs that still say `EDA_02_outputs`) → they're historical and will refresh on the next execution.

---

## Suggested order if completing today

1. **Update `CLAUDE.md` §8** — 2 minutes; commits with this action plan.
2. **Add the superseded note** to `project_review.md` — 1 minute.
3. **Stop unless adult BraTS data arrives.** The project is shippable.

If/when adult BraTS data arrives, follow TASK 9's plan above.

---

## Summary

The project is **functionally complete**. Every spec-required component (Dice, IoU, HD95, 3-way model comparison, 3D test evaluation, post-processing, clinical NIfTI export, unit tests) is implemented, tested, documented, and committed to `origin/master`. The only remaining roadmap item is **TASK 9 (cross-domain)**, gated on external data.

- **Critical** (must do for spec compliance): none.
- **Strongly recommended**: TASK 9 (if adult BraTS data available).
- **Nice-to-have**: `.venv-1`/`.venv-2` cleanup; `CLAUDE.md` §8 refresh; `project_review.md` "superseded" note.
- **Stretch**: integration test for the full eval pipeline.

After completing the optional `CLAUDE.md` + `project_review.md` polish, the project moves from ~99% → ~100% relative to its original specification. TASK 9 adds a generalisation experiment but is not part of the original deliverable.
