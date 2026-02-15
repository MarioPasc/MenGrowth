# Step 1: Data Harmonization

## Theory

Data harmonization converts heterogeneous raw MRI files into a standardized format suitable for downstream processing. This step performs three sub-operations:

1. **Format conversion (NRRD → NIfTI):** Clinical MRI data often arrives in NRRD format (Nearly Raw Raster Data). NIfTI (Neuroimaging Informatics Technology Initiative) is the standard for neuroimaging pipelines, supported by ANTs, FreeSurfer, FSL, and deep learning frameworks. The conversion preserves voxel data, affine matrix, and spatial metadata while switching to a format with broader tool compatibility.

2. **Reorientation to canonical axes:** MRI scanners may store volumes in arbitrary orientations (LPS, RAS, etc.). Reorienting all volumes to a single convention (default: RAS — Right-Anterior-Superior) ensures consistent spatial interpretation across patients and simplifies subsequent registration and visualization.

3. **Background zeroing:** Raw MRI volumes contain non-zero background signal (noise outside the head). Zeroing this background improves registration convergence, reduces computational waste, and prevents background noise from corrupting intensity statistics. Three algorithms are available:
   - **Border-connected percentile:** Conservative approach using percentile thresholding + connected components from image borders
   - **SELF head mask:** Sophisticated air-head separation using flood-fill from dark voxel seeds (Beul et al.)
   - **Otsu foreground:** Otsu's automatic thresholding with morphological cleanup (recommended for robustness)

## Motivation

Meningioma datasets from multiple institutions arrive in NRRD with inconsistent orientations (LPS from some scanners, RAS from others). Without harmonization, registration algorithms fail silently or produce incorrect transforms. Background noise at MRI-typical levels (Rician distribution) biases intensity normalization statistics and creates spurious features for deep learning models.

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/data_harmonization.py`
- **Format converter:** `mengrowth/preprocessing/src/data_harmonization/io.py` → `NRRDtoNIfTIConverter`
- **Reorienter:** `mengrowth/preprocessing/src/data_harmonization/orient.py` → `Reorienter`
- **Background removal:**
  - `mengrowth/preprocessing/src/data_harmonization/head_masking/conservative.py` → `ConservativeBackgroundRemover`
  - `mengrowth/preprocessing/src/data_harmonization/head_masking/self.py` → `SELFBackgroundRemover`
  - `mengrowth/preprocessing/src/data_harmonization/head_masking/otsu_foreground.py` → `OtsuForegroundRemover`
- **Low-level NRRD parsing:** `mengrowth/preprocessing/src/utils/nrrd_to_nifti.py` → `nifti_write_3d()`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `DataHarmonizationStepConfig`, `BackgroundZeroingConfig`
- **YAML key:** `step_configs.data_harmonization`

## Config Reference

```yaml
data_harmonization:
  save_visualization: true        # Generate before/after PNG
  reorient_to: "RAS"              # Target orientation ("RAS" or "LPS")
  background_zeroing:
    method: "otsu_foreground"     # "border_connected_percentile" | "self_head_mask" | "otsu_foreground" | null
    # Otsu parameters (recommended method)
    gaussian_sigma_px: 1.0        # Smoothing before thresholding
    min_component_voxels: 1000    # Minimum foreground component size
    n_components_to_keep: 1       # Number of largest components to retain
    air_border_margin: 1          # Erosion iterations on air mask (more conservative)
    expand_air_mask: 0            # Dilation iterations on air mask (less conservative)
```

## Inputs / Outputs

- **Input:** `{dataset_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c.nrrd`
- **Output:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1c.nii.gz`
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step1_data_harmonization_t1c.png`
- **Execution level:** per-modality

## Common Tasks

| Task | How |
|------|-----|
| Change orientation convention | Set `reorient_to: "LPS"` in config |
| Disable background removal | Set `background_zeroing.method: null` |
| Switch background method | Change `method` to `"self_head_mask"`, `"border_connected_percentile"`, or `"otsu_foreground"` |
| Make background removal more aggressive | Increase `expand_air_mask` or decrease `air_border_margin` |
| Debug background removal | Check visualization PNGs — they show before/after with masked regions highlighted |
