# 3D NIfTI Predictions — Clinical Inspection

Six post-processed 3D predictions: each of the three models contributes its **best-** and **worst-Dice** test subject (ranked by mean foreground Dice across NCR + ED + ET).

Open each `.nii.gz` in **3D Slicer** or **ITK-SNAP** alongside the matching raw modalities at
`PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training/{subject_id}/{subject_id}-{t1c,t1n,t2f,t2w}.nii.gz`
to visually inspect contour quality at the tumour boundary.

## Picks

| Model         | Best subject            | Best Dice | Worst subject           | Worst Dice |
|---------------|-------------------------|-----------|-------------------------|------------|
| U-Net         | BraTS-PED-00229-000     | 0.9865    | BraTS-PED-00017-000     | 0.0892     |
| FPN           | BraTS-PED-00229-000     | 0.9770    | BraTS-PED-00017-000     | 0.0667     |
| SegFormer-B1  | BraTS-PED-00234-000     | 0.9849    | BraTS-PED-00017-000     | 0.1250     |

All three models agree on the worst subject (BraTS-PED-00017-000); U-Net and FPN agree on the best (BraTS-PED-00229-000), while SegFormer-B1's best is BraTS-PED-00234-000.

## File format

- **Shape:** `(240, 240, 155)` int16 (full BraTS-PEDs canvas, no crop).
- **Labels:** `0` BG · `1` NCR · `2` ED · `3` ET (standard BraTS-PEDs convention).
- **Affine & header:** copied from each subject's reference `-seg.nii.gz`, so the predictions are spatially aligned with the raw scans and can be overlaid directly.
- **Post-processing:** `remove_small_components(min_voxels=50)` with 26-connectivity (kills the "confetti effect" from 2D slice-based predictions).

## How they were generated

```bash
python -m src.export_nifti_predictions
```

This re-runs `predict_volume()` on the test set, picks best/worst per model from the live JSONs in `evaluation_outputs/test_3d_metrics_*.json`, and saves the post-processed predictions here. Reproducible — outputs are deterministic given the committed checkpoints.
