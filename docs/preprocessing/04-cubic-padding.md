# Step 4: Cubic Padding

## Theory

After resampling to isotropic resolution, volumes from different modalities within the same study may still have different matrix sizes (e.g., T1 is 256×256×192 while FLAIR is 256×256×160). Cubic padding zero-pads all volumes to a cubic shape (N×N×N) by symmetrically adding slices along each axis to match the largest dimension.

The padding operation:
1. Finds the maximum dimension across all modalities in the study (or per individual image)
2. Computes symmetric padding: `pad = (max_dim - current_dim) // 2` per axis
3. Fills padded regions with either the image minimum intensity or zero
4. Centers the brain within the cubic volume

This is a study-level operation because the target cubic size should be consistent across all modalities within a study to maintain spatial correspondence.

## Motivation

Registration algorithms (ANTs) and deep learning models require consistent input dimensions. Without cubic padding:
- Affine rotations during registration may clip brain regions at volume edges
- Boundary artifacts appear when transforms move tissue outside the FOV
- Different modalities with different FOV extents cannot be directly compared voxel-wise

Symmetric padding is preferred over asymmetric because it preserves the brain's position near the volume center, which aids center-of-mass initialization in registration.

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/cubic_padding.py`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `CubicPaddingStepConfig`, `CubicPaddingConfig`
- **YAML key:** `step_configs.cubic_padding`

## Config Reference

```yaml
cubic_padding:
  save_visualization: true
  cubic_padding:
    method: "symmetric"                      # "symmetric" | null (null to skip)
    fill_value_mode: "min"                   # "min" (image minimum) | "zero"
    target_shape_mode: "max_across_modalities"  # "max_across_modalities" | "max_per_modality"
```

**Parameter rationale:**
- `fill_value_mode="min"`: Using the image minimum (typically 0 after background zeroing) avoids introducing artificial intensity discontinuities at padding boundaries
- `target_shape_mode="max_across_modalities"`: Ensures all modalities in a study share the same matrix size, critical for multi-modal registration

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nii.gz` (all modalities in study)
- **Output:** Same paths (in-place update via temp files)
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step4_cubic_padding_*.png`
- **Execution level:** per-study

## Common Tasks

| Task | How |
|------|-----|
| Skip padding | Set `method: null` |
| Pad each modality independently | Set `target_shape_mode: "max_per_modality"` |
| Use zero padding instead of min | Set `fill_value_mode: "zero"` |
| Debug padding issues | Check visualization PNGs — they show volume extent before/after |
