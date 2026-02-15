# Preprocessing Rules

## Step Execution Levels

Every preprocessing step operates at one of three levels. When implementing or modifying a step, match the correct level:

| Level | Scope | Handler signature receives |
|-------|-------|---------------------------|
| `modality` | Per (patient, study, modality) file | `study_dir`, `modality`, `paths` |
| `study` | Per (patient, study) | `study_dir`, all modalities in study |
| `patient` | Per patient across all studies | `all_study_dirs` |

The level is declared in `STEP_METADATA` in `mengrowth/preprocessing/src/config.py`.

## Temp-File Pattern

All file writes must use the temp-file pattern to prevent partial writes from corrupting the dataset:

```python
temp_path = output_path.with_suffix('.tmp.nii.gz')
# ... write processing result to temp_path ...
temp_path.rename(output_path)  # Atomic replacement
```

Never write directly to the final output path.

## Artifact Naming

Intermediate artifacts saved to `{preprocessing_artifacts_path}` must follow these conventions:

| Artifact | Pattern | Example |
|----------|---------|---------|
| Brain mask | `{modality}_brain_mask.nii.gz` | `t1c_brain_mask.nii.gz` |
| Bias field | `{modality}_bias_field.nii.gz` | `t1c_bias_field.nii.gz` |
| Transform | Step-specific (`.h5`, `.mat`) | Registration transforms |

Artifacts are stored at: `{artifacts}/{patient_id}/{study_id}/`

## Visualization Naming

Step visualizations saved to `{viz_root}` follow:

```
step{N}_{step_name}_{modality}.png
```

Example: `step2_bias_field_correction_t1c.png`

## Config-to-Step Mapping

Each step in the `steps:` YAML list must have a matching entry in `step_configs:`. The `StepRegistry` uses substring matching — `"intensity_normalization"` matches both `"intensity_normalization"` and `"intensity_normalization_2"`.

When adding a new step:
1. Add `StepMetadata` entry in `src/config.py`
2. Add step config `@dataclass` in `src/config.py`
3. Add step config mapping in `_convert_step_configs()`
4. Create handler function in `src/steps/`
5. Register handler in `preprocess.py` → `_register_step_handlers()`

## Never Modify Raw Data

The preprocessing pipeline operates on copies in `{output_root}` (test mode) or updates in-place within the curated dataset (pipeline mode). The original raw data directories must never be modified. The `.claude/settings.json` deny rules enforce this for the raw data path.

## Brain Mask Resolution

When a step needs a brain mask (e.g., intensity normalization):
1. Check `{artifacts}/{patient_id}/{study_id}/{modality}_brain_mask.nii.gz`
2. If not found, fall back to `image > 0` (nonzero voxels)
3. Always log which mask source was used

## Config Dataclass Conventions

- Every config field must have a default value (no required fields)
- Use `__post_init__()` for validation
- Support `dict` → `dataclass` conversion (for YAML parsing)
- Include backwards compatibility aliases where needed (e.g., `Step0DataHarmonizationConfig`)
- All config objects must be picklable (no lambdas, no open file handles)
