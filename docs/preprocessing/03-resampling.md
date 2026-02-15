# Step 3: Resampling

## Theory

Clinical MRI acquisitions have heterogeneous voxel spacing — a T1 might be acquired at 0.5×0.5×1.0 mm while a FLAIR from the same study is 0.9×0.9×5.0 mm. Resampling standardizes all volumes to a uniform isotropic resolution (default: 1.0×1.0×1.0 mm), enabling:
- Voxel-wise operations across modalities and timepoints
- Consistent spatial resolution for deep learning input
- Fair comparison of volumetric measurements

Three resampling methods are available:

1. **BSpline interpolation** (default): Classical cubic B-spline interpolation (order 3). Fast, deterministic, and sufficient for moderate upsampling ratios (≤2×). Produces smooth results but may blur fine structures at large upsampling factors.

2. **ECLARE** (deep learning): Edge-Conditioned Learned Arbitrary-resolution REsampling. A deep learning super-resolution method that preserves structural edges during large-factor upsampling. Requires a separate conda environment with ECLARE installed and GPU resources.

3. **Composite**: Hybrid strategy that automatically selects between BSpline and ECLARE per-dimension based on the upsampling factor:
   - Dimensions within `max_mm_interpolator` of target: BSpline only
   - Dimensions between `max_mm_interpolator` and `max_mm_dl_method`: ECLARE
   - Dimensions beyond `max_mm_dl_method`: BSpline downsample first, then ECLARE

An optional pre-resampling intensity normalization can be applied (configured within the resampling step) to stabilize intensity ranges before interpolation.

## Motivation

Clinical meningioma MRI has extreme voxel anisotropy — through-plane spacing of 5–7 mm is common for 2D acquisitions, while in-plane resolution is 0.5–1.0 mm. Without resampling, 3D convolutions see very different physical extents per voxel, degrading segmentation accuracy. Isotropic 1 mm³ is the BraTS convention and balances resolution with computational cost.

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/resampling.py`
- **BSpline:** `mengrowth/preprocessing/src/resampling/bspline.py` → `BSplineResampler`
- **ECLARE:** `mengrowth/preprocessing/src/resampling/eclare.py` → `EclareResampler`
- **Composite:** `mengrowth/preprocessing/src/resampling/composite.py` → `CompositeResampler`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `ResamplingStepConfig`, `ResamplingConfig`
- **YAML key:** `step_configs.resampling`

## Config Reference

```yaml
resampling:
  save_visualization: true
  resampling:
    method: "bspline"             # "bspline" | "eclare" | "composite" | null
    target_voxel_size: [1.0, 1.0, 1.0]  # Target isotropic spacing in mm
    bspline_order: 3              # Interpolation order [0-5]

    # Pre-resampling normalization (optional)
    normalize_method: null        # "zscore" | "kde" | "percentile_minmax" | null

    # ECLARE parameters (if method == "eclare")
    conda_environment_eclare: "eclare_env"
    batch_size: 128
    gpu_id: 0

    # Composite parameters (if method == "composite")
    max_mm_interpolator: 1.2      # Max spacing for BSpline-only
    max_mm_dl_method: 5.0         # Max spacing for ECLARE
    resample_mm_to_interpolator_if_max_mm_dl_method: 3.0
```

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c.nii.gz` (bias-corrected)
- **Output:** Same path (in-place update via temp file)
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step3_resampling_t1c.png`
- **Execution level:** per-modality

## Common Tasks

| Task | How |
|------|-----|
| Change target resolution | Modify `target_voxel_size` (e.g., `[0.5, 0.5, 0.5]` for higher resolution) |
| Use deep learning upsampling | Set `method: "eclare"` and configure `conda_environment_eclare` |
| Use hybrid strategy | Set `method: "composite"` and tune `max_mm_interpolator` / `max_mm_dl_method` |
| Skip resampling | Set `method: null` |
| Add pre-resampling normalization | Set `normalize_method: "zscore"` (or other method) |
