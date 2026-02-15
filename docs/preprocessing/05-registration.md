# Step 5: Registration

## Theory

Registration spatially aligns images into a common coordinate frame by computing geometric transforms that maximize similarity between a moving and fixed image. This step performs two sub-registrations:

### 5a: Intra-Study Multi-Modal Coregistration

Within each study (timepoint), different MRI sequences (T1c, T1n, T2w, T2f) are acquired at slightly different positions due to patient motion between scans. Coregistration aligns all modalities to a reference modality (default priority: T1c > T1n > T2f > T2w).

The ANTs registration framework optimizes a similarity metric (Mattes Mutual Information for multi-modal) over a parametric transform space:

- **Rigid** (6 DOF): rotation + translation. Sufficient for within-session motion.
- **Affine** (12 DOF): adds scaling + shearing. Handles minor geometric distortions.
- **SyN** (diffeomorphic): non-linear deformation. Overkill for intra-study but available.

Multi-resolution optimization progressively refines transforms from coarse (8× downsampled) to native resolution, avoiding local minima.

### 5b: Atlas Registration

After intra-study coregistration, the reference modality is registered to a standard atlas (e.g., SRI24 T1 template). The resulting transform is then propagated to all other modalities via composition, bringing the entire study into atlas space. This enables:
- Population-level comparisons in a common coordinate frame
- Consistent anatomical landmarks across patients
- Template-based analysis and reporting

Two registration engines are available: **Nipype** (ANTs command-line wrapper) and **ANTsPyx** (Python-native ANTs bindings). ANTsPyx is faster for single registrations but Nipype provides more detailed logging.

## Motivation

Meningioma growth analysis requires voxel-wise comparison across timepoints and modalities. Without registration:
- Multi-modal features cannot be extracted at the same anatomical location
- Longitudinal volume measurements are confounded by different head positions
- Atlas-based analyses (lobe assignment, anatomical reporting) are impossible

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/registration.py`
- **Intra-study coregistration:**
  - `mengrowth/preprocessing/src/registration/multi_modal_coregistration.py` (Nipype)
  - `mengrowth/preprocessing/src/registration/antspyx_multi_modal_coregistration.py` (ANTsPyx)
- **Atlas registration:**
  - `mengrowth/preprocessing/src/registration/intra_study_to_atlas.py` (Nipype)
  - `mengrowth/preprocessing/src/registration/antspyx_intra_study_to_atlas.py` (ANTsPyx)
- **Registration factory:** `mengrowth/preprocessing/src/registration/factory.py`
- **Diagnostic parser:** `mengrowth/preprocessing/src/registration/diagnostic_parser.py`
- **Stdout capture:** `mengrowth/preprocessing/src/registration/stdout_capture.py`
- **Constants:** `mengrowth/preprocessing/src/registration/constants.py`
- **Config classes:** `mengrowth/preprocessing/src/config.py` → `RegistrationStepConfig`, `IntraStudyToReferenceConfig`, `IntraStudyToAtlasConfig`
- **YAML key:** `step_configs.registration_static` (or whatever name used in `steps:` list)

## Config Reference

```yaml
registration_static:
  save_visualization: true
  intra_study_to_reference:
    method: "ants"                  # "ants" | null
    engine: "antspyx"               # "nipype" | "antspyx"
    reference_modality_priority: "t1c > t1n > t2f > t2w"
    transform_type: "Affine"        # "Rigid" | "Affine" | "SyN" | list
    metric: "MI"                    # "Mattes" | "MI" | "CC" | "MeanSquares"
    metric_bins: 64
    sampling_strategy: "Random"
    sampling_percentage: 0.4
    number_of_iterations: [[1000, 500, 250]]
    shrink_factors: [[4, 2, 1]]
    smoothing_sigmas: [[2, 1, 0]]
    convergence_threshold: 1.0e-6
    interpolation: "BSpline"
    use_center_of_mass_init: true
    validate_registration_quality: true
    quality_warning_threshold: -0.3

  intra_study_to_atlas:
    method: "ants"
    engine: "antspyx"
    atlas_path: "/path/to/SRI24_T1.nii"
    transforms: ["Rigid", "Affine"]
    metric: "MI"
    metric_bins: 128
    sampling_percentage: 0.5
    number_of_iterations: [[1000, 500, 250], [500, 250, 100]]
    shrink_factors: [[4, 2, 1], [2, 1, 1]]
    smoothing_sigmas: [[2, 1, 0], [1, 0, 0]]
```

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nii.gz` (padded volumes)
- **Output:** Same paths (registered, in atlas space)
- **Artifacts:** Transform files (`.h5`, `.mat`) in `{artifacts}/`
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step5_registration_*.png`
- **Execution level:** per-study

## Common Tasks

| Task | How |
|------|-----|
| Skip atlas registration | Set `intra_study_to_atlas.method: null` |
| Switch to Nipype engine | Set `engine: "nipype"` (both sub-steps) |
| Change reference modality | Modify `reference_modality_priority` string |
| Use more aggressive transform | Set `transform_type: ["Rigid", "Affine", "SyN"]` |
| Debug registration quality | Check quality warning logs; set `validate_registration_quality: true` |
| Inspect transform files | Look in `{artifacts}/` for `.h5` composite transforms |
