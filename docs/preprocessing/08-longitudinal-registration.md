# Step 8: Longitudinal Registration

## Theory

Longitudinal registration aligns MRI volumes acquired at different timepoints for the same patient into a common spatial frame, enabling voxel-wise comparison of disease progression. This is distinct from intra-study registration (Step 5) which aligns modalities within a single timepoint.

The algorithm:
1. **Select a reference timestamp** — either from a per-patient YAML mapping, or automatically via quality-based selection (evaluating SNR, CNR, and boundary gradient scores)
2. **Register all other timestamps to the reference** — using the same ANTs framework as Step 5, but optimized for same-modality comparison (typically higher sampling percentage and Mattes MI metric)
3. **Optionally propagate brain masks** — warp the reference brain mask to each timestamp for QC (Dice comparison with independently computed masks)

Two registration modes are supported:
- **Single-reference mode** (default): Select one modality across all timestamps (e.g., always register T1n), apply the same transform to other modalities
- **Per-modality mode**: Register each modality independently across timestamps (more transforms but avoids cross-modal registration artifacts)

### Reference Selection

Automatic reference selection uses quality metrics to choose the "best" timestamp:
1. Compute per-timestamp quality scores (SNR, CNR, boundary gradient)
2. Rank timestamps by composite quality
3. Select the highest-quality timestamp (with optional preference for earlier timepoints)
4. Optionally validate with Jacobian determinant statistics (flagging excessive deformation)

## Motivation

Meningioma growth prediction is fundamentally a longitudinal analysis — comparing tumor size, shape, and signal characteristics across timepoints. Without longitudinal registration:
- Volume change measurements include head repositioning artifacts
- Voxel-wise growth maps are meaningless
- Temporal analysis of signal evolution is confounded by geometric differences

The quality-based reference selection avoids registration bias from low-quality acquisitions pulling higher-quality timepoints toward a degraded space.

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/longitudinal_registration.py`
- **Registration implementation:** `mengrowth/preprocessing/src/registration/longitudinal_registration.py`
- **Reference selection:** `mengrowth/preprocessing/src/registration/reference_selection.py`
- **ANTsPyx implementation:** `mengrowth/preprocessing/src/registration/antspyx_multi_modal_coregistration.py` (reused)
- **Config class:** `mengrowth/preprocessing/src/config.py` → `LongitudinalRegistrationStepConfig`, `LongitudinalRegistrationConfig`
- **YAML key:** `step_configs.longitudinal_registration`

## Config Reference

```yaml
longitudinal_registration:
  save_visualization: true
  longitudinal_registration:
    method: "ants"                        # "ants" | null
    engine: "antspyx"                     # "nipype" | "antspyx"
    reference_modality_priority: "t1n > t1c > t2f > t2w"  # or "per_modality"
    reference_timestamp_per_study: null   # Path to YAML with patient→timestamp mapping

    # Automatic reference selection
    reference_selection_method: "quality_based"  # "quality_based" | "first" | "last" | "midpoint"
    reference_selection_metrics: ["snr_foreground", "cnr_high_low", "boundary_gradient_score"]
    reference_selection_prefer_earlier: true
    reference_selection_validate_jacobian: true
    reference_selection_jacobian_threshold: 0.5

    # Transform parameters
    transform_type: ["Rigid", "Affine"]
    metric: "Mattes"
    metric_bins: 64
    sampling_strategy: "Random"
    sampling_percentage: 0.5

    # Multi-resolution schedule
    number_of_iterations: [[1000, 500, 250, 0], [1000, 500, 250, 0]]
    shrink_factors: [[8, 4, 2, 1], [8, 4, 2, 1]]
    smoothing_sigmas: [[3, 2, 1, 0], [3, 2, 1, 0]]

    convergence_threshold: 1.0e-6
    interpolation: "BSpline"

    # Mask propagation for QC
    propagate_reference_mask: false
    compute_mask_comparison: false
```

## Inputs / Outputs

- **Input:** All studies for a patient: `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-*/`
- **Output:** Same paths (all timestamps registered to reference timestamp)
- **Artifacts:** Transform files, optional propagated masks in `{artifacts}/`
- **Visualization:** `{viz_root}/MenGrowth-XXXX/longitudinal_registration_*.png`
- **Execution level:** per-patient

## Common Tasks

| Task | How |
|------|-----|
| Specify reference timestamp manually | Create YAML mapping `patient_id: timestamp_dir_name` and set `reference_timestamp_per_study` |
| Use earliest timepoint as reference | Set `reference_selection_method: "first"` |
| Register each modality independently | Set `reference_modality_priority: "per_modality"` |
| Validate with mask propagation | Set `propagate_reference_mask: true` and `compute_mask_comparison: true` |
| Skip longitudinal registration | Set `method: null` or remove from `steps:` list |
| Debug registration quality | Check Jacobian threshold warnings in logs; examine visualization PNGs |
