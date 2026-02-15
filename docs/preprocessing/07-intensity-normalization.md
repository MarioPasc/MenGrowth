# Step 7: Intensity Normalization

## Theory

MRI intensities are arbitrary — the same tissue type can produce vastly different signal values across scanners, protocols, and even repeat acquisitions on the same scanner. Intensity normalization maps these arbitrary values to a standardized scale, enabling consistent intensity-based analysis across the dataset.

Six normalization methods are available, each with different assumptions:

### Z-Score (nnU-Net/BraTS convention)

```
I'(x) = (I(x) - μ_brain) / σ_brain
```

Computes mean and standard deviation over brain voxels only (mask > 0). Background remains zero. Optionally clipped to `[low, high]` range. This is the default for BraTS/nnU-Net pipelines and makes no tissue-specific assumptions.

### KDE (Kernel Density Estimation)

Estimates the intensity distribution via kernel density estimation, identifies the mode (most frequent intensity = white matter peak), and normalizes relative to it. Requires `intensity-normalization` package.

### Percentile Min-Max

```
I'(x) = (I(x) - P_low) / (P_high - P_low)
```

Clips to `[p1, p2]` percentile range (default: 1st–99th), then scales to `[0, 1]`. Simple and robust to outliers. Computed on brain voxels only.

### WhiteStripe

Identifies the white matter peak in the intensity histogram using kernel density estimation, then standardizes intensities relative to that peak. Specifically designed for T1-weighted images. Requires `intensity-normalization` package.

### FCM (Fuzzy C-Means)

Segments brain tissue into clusters (default: 3 for WM/GM/CSF), identifies the target tissue cluster (default: WM), and normalizes by its mean intensity. Makes explicit tissue-type assumptions. Requires `intensity-normalization` package.

### LSQ (Least Squares)

Performs piecewise linear normalization by matching histogram landmarks (deciles) to a standard scale using least squares fitting. Requires `intensity-normalization` package.

**All normalizers use brain masks** — either from skull stripping artifacts (`{modality}_brain_mask.nii.gz`) or a fallback of nonzero voxels (`image > 0`).

## Motivation

Multi-institutional meningioma data spans different scanner manufacturers (Siemens, GE, Philips), field strengths (1.5T, 3T), and protocols. Without normalization:
- Deep learning models overfit to site-specific intensity profiles
- Longitudinal growth measurements are confounded by acquisition parameter changes
- Population-level statistics are meaningless

Z-score is recommended as the default for deep learning workflows (BraTS/nnU-Net standard). Tissue-specific methods (WhiteStripe, FCM) may be preferable for statistical analyses where white matter should be at a fixed reference value.

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/intensity_normalization.py`
- **Normalizer implementations:**
  - `mengrowth/preprocessing/src/normalization/zscore.py` → `ZScoreNormalizer`
  - `mengrowth/preprocessing/src/normalization/kde.py` → `KDENormalizer`
  - `mengrowth/preprocessing/src/normalization/percentile_minmax.py` → `PercentileMinMaxNormalizer`
  - `mengrowth/preprocessing/src/normalization/whitestripe.py` → `WhiteStripeNormalizer`
  - `mengrowth/preprocessing/src/normalization/fcm.py` → `FCMNormalizer`
  - `mengrowth/preprocessing/src/normalization/lsq.py` → `LSQNormalizer`
- **Base class:** `mengrowth/preprocessing/src/normalization/base.py` → `BaseNormalizer`
- **Utilities:** `mengrowth/preprocessing/src/normalization/utils.py`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `IntensityNormalizationStepConfig`, `IntensityNormalizationConfig`
- **YAML key:** `step_configs.intensity_normalization`

## Config Reference

```yaml
intensity_normalization:
  save_visualization: true
  intensity_normalization:
    method: "zscore"              # "zscore" | "kde" | "percentile_minmax" | "whitestripe" | "fcm" | "lsq" | null

    # Z-score parameters
    norm_value: 1.0               # Post-normalization scaling factor
    clip_range: null              # Optional [low, high] clipping (e.g., [-5.0, 5.0])

    # Percentile MinMax parameters
    p1: 1.0                       # Lower percentile
    p2: 99.0                      # Upper percentile

    # WhiteStripe parameters
    width: 0.05                   # Quantile range for WM peak detection
    width_l: null                 # Optional lower bound override
    width_u: null                 # Optional upper bound override

    # FCM parameters
    n_clusters: 3                 # Number of tissue clusters
    tissue_type: "WM"             # Target tissue: "WM" | "GM" | "CSF"
    max_iter: 50                  # Max FCM iterations
    error_threshold: 0.005        # Convergence threshold
    fuzziness: 2.0                # Cluster membership fuzziness
```

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c.nii.gz` (skull-stripped)
- **Output:** Same path (in-place update via temp file)
- **Brain mask resolution:** `{artifacts}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/{modality}_brain_mask.nii.gz` (from skull stripping)
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step7_intensity_normalization_t1c.png`
- **Execution level:** per-modality
- **Returns:** Dict with `mean`, `std`, `brain_voxel_count`, `brain_coverage_percent`, `mask_source`, intensity ranges

## Common Tasks

| Task | How |
|------|-----|
| Use BraTS-standard normalization | Set `method: "zscore"` |
| Clip extreme z-scores | Set `clip_range: [-5.0, 5.0]` |
| Use tissue-specific normalization | Set `method: "fcm"` with `tissue_type: "WM"` |
| Run normalization twice | Add `intensity_normalization_2` to `steps:` with separate config |
| Skip normalization | Set `method: null` |
| Debug normalization | Check visualization PNG showing before/after histograms |
