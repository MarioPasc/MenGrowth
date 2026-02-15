# Step 2: Bias Field Correction

## Theory

MRI images suffer from intensity inhomogeneity (bias field) — a smooth, low-frequency spatial variation in signal intensity caused by RF coil sensitivity patterns, eddy currents, and patient anatomy interactions. The bias field manifests as gradual intensity gradients across the image: the same tissue type appears brighter near the coil and darker far from it.

The N4ITK algorithm (Tustison et al., 2010) models the observed image as:

```
I_observed(x) = I_true(x) × B(x) + N(x)
```

where `B(x)` is the multiplicative bias field and `N(x)` is additive noise. N4 estimates `B(x)` iteratively using B-spline fitting in log-space, then divides it out:

```
I_corrected(x) = I_observed(x) / B(x)
```

N4 improves on the earlier N3 algorithm by using a more robust convergence criterion and multi-resolution optimization with configurable shrink factors. The algorithm operates in log-space where the multiplicative bias becomes additive, fitting a B-spline approximation to the residual field.

## Motivation

Bias field artifacts are ubiquitous in clinical MRI, especially at 1.5T and 3T field strengths used for meningioma imaging. Without correction:
- Intensity-based segmentation algorithms misclassify tissue boundaries
- Intensity normalization statistics are skewed by spatial position
- Registration similarity metrics (MI, NCC) are degraded by position-dependent intensity shifts
- Longitudinal growth measurements may be confounded by varying coil positioning

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/bias_field_correction.py`
- **N4 implementation:** `mengrowth/preprocessing/src/bias_field_correction/n4_sitk.py` → `N4BiasFieldCorrector`
- **Utility wrapper:** `mengrowth/preprocessing/src/utils/n4_bias_fields.py`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `BiasFieldCorrectionStepConfig`, `BiasFieldCorrectionConfig`
- **YAML key:** `step_configs.bias_field_correction`

## Config Reference

```yaml
bias_field_correction:
  save_visualization: true        # Generate before/after PNG
  save_artifact: true             # Save estimated bias field to artifacts/
  bias_field_correction:
    method: "n4"                  # "n4" | null (null to skip)
    shrink_factor: 4              # Downsampling factor for speed [1-8]
    max_iterations: [50,50,50,50] # Iterations per resolution level (4 levels)
    bias_field_fwhm: 0.15         # FWHM of Gaussian smoothing [0.01-1.0]
    convergence_threshold: 0.001  # Early stopping threshold
```

**Key parameter rationale:**
- `shrink_factor=4`: Processing at 4× downsampled resolution is standard for N4. The bias field is smooth enough that high resolution is unnecessary, and this reduces computation by 64×.
- `max_iterations=[50,50,50,50]`: Four resolution levels with 50 iterations each. The multi-resolution approach prevents local minima.
- `bias_field_fwhm=0.15`: Controls smoothness of the estimated field. Lower values allow sharper corrections but risk fitting noise.

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c.nii.gz`
- **Output:** Same path (in-place update via temp file)
- **Artifact:** `{artifacts}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c_bias_field.nii.gz`
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step2_bias_field_correction_t1c.png`
- **Execution level:** per-modality

## Common Tasks

| Task | How |
|------|-----|
| Skip bias correction | Set `method: null` in config |
| Increase correction strength | Reduce `convergence_threshold` or increase `max_iterations` |
| Speed up correction | Increase `shrink_factor` (max 8) |
| Inspect estimated bias field | Check `{artifacts}/.../t1c_bias_field.nii.gz` — smooth spatial map |
| Debug poor correction | Visualization PNG shows before/after intensity profiles; verify input has background zeroed |
