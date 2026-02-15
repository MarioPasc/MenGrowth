# Preprocessing Pipeline Overview

## Purpose

The BraTS-like preprocessing pipeline transforms curated NRRD datasets into analysis-ready NIfTI volumes suitable for deep learning training and longitudinal growth prediction. It standardizes geometry, intensity, and spatial alignment across all patients and timepoints.

## Pipeline Diagram

```
Curated NRRD Dataset (from curation pipeline)
  │
  ├─ [1. Data Harmonization]     NRRD→NIfTI, reorient to RAS, background zeroing
  ├─ [2. Bias Field Correction]  N4ITK inhomogeneity correction
  ├─ [3. Resampling]             Isotropic voxel spacing (BSpline/ECLARE/Composite)
  ├─ [4. Cubic Padding]          Zero-pad to cubic FOV for registration stability
  ├─ [5. Registration]           Intra-study coregistration + atlas alignment
  ├─ [6. Skull Stripping]        Brain extraction (HD-BET / SynthStrip)
  ├─ [7. Intensity Normalization] Z-score / KDE / FCM / WhiteStripe / PercentileMinMax / LSQ
  └─ [8. Longitudinal Registration] Cross-timepoint alignment within patient
  │
  ▼
Analysis-Ready NIfTI Dataset
```

Steps are defined in YAML config (`steps:` list) and can be reordered, repeated with different configs (e.g., `intensity_normalization_1`, `intensity_normalization_2`), or omitted.

## Architecture

### Orchestrator Pattern

`PreprocessingOrchestrator` (in `mengrowth/preprocessing/src/preprocess.py`) drives execution:

1. Reads the `steps:` list from config
2. For each step, resolves the handler via `StepRegistry` (substring matching)
3. Determines execution level from `STEP_METADATA` (modality / study / patient)
4. Iterates over the appropriate scope and calls the step handler

### Execution Levels

| Level | Scope | Steps |
|-------|-------|-------|
| **modality** | Called once per `(patient, study, modality)` | Data harmonization, bias field correction, resampling, intensity normalization |
| **study** | Called once per `(patient, study)` | Cubic padding, registration, skull stripping |
| **patient** | Called once per patient (all studies) | Longitudinal registration |

### Step Registry

Steps are registered by pattern name. The registry uses substring matching:
- `"intensity_normalization"` matches `"intensity_normalization"`, `"intensity_normalization_2"`, etc.
- Order in `STEP_METADATA` matters: more specific patterns (`"longitudinal_registration"`) must precede general ones (`"registration"`)

### Key Classes

| Class | File | Role |
|-------|------|------|
| `PreprocessingOrchestrator` | `preprocessing/src/preprocess.py` | Pipeline execution coordinator |
| `StepRegistry` | `preprocessing/src/config.py` | Pattern→handler mapping |
| `StepExecutionContext` | `preprocessing/src/config.py` | Per-invocation context passed to handlers |
| `StepMetadata` | `preprocessing/src/config.py` | Declares execution level per step type |
| `PipelineExecutionConfig` | `preprocessing/src/config.py` | Top-level config with `steps`, `step_configs`, paths |
| `BasePreprocessingStep` | `preprocessing/src/base.py` | ABC with `execute()` + `visualize()` |

## Entry Point

```bash
mengrowth-preprocess --config configs/icaiserver/preprocessing_icai.yaml [--patient MenGrowth-0015] [--verbose] [--dry-run]
```

**CLI:** `mengrowth/cli/preprocess.py`

## Config Structure

```yaml
preprocessing:
  patient_selector: "single"  # or "all"
  patient_id: "MenGrowth-0001"
  mode: "test"                # "test" (separate output) or "pipeline" (in-place)
  dataset_root: "/path/to/curated/dataset"
  output_root: "/path/to/output"
  viz_root: "/path/to/visualizations"
  preprocessing_artifacts_path: "/path/to/artifacts"
  modalities: ["t1c", "t1n", "t2w", "t2f"]

  steps:
    - data_harmonization
    - bias_field_correction
    - resampling
    - cubic_padding
    - registration_static
    - skull_stripping
    - intensity_normalization
    - longitudinal_registration

  step_configs:
    data_harmonization: { ... }
    bias_field_correction: { ... }
    # ... one config block per step pattern
```

**Config file:** `configs/icaiserver/preprocessing_icai.yaml`
**Config parser:** `mengrowth/preprocessing/src/config.py` → `load_preprocessing_pipeline_config()`

## Output Structure

```
{output_root}/
  MenGrowth-XXXX/
    MenGrowth-XXXX-YYY/
      t1c.nii.gz, t1n.nii.gz, t2w.nii.gz, t2f.nii.gz

{viz_root}/
  MenGrowth-XXXX/
    MenGrowth-XXXX-YYY/
      step1_data_harmonization_t1c.png
      step2_bias_field_correction_t1c.png
      ...

{artifacts}/
  MenGrowth-XXXX/
    MenGrowth-XXXX-YYY/
      t1c_bias_field.nii.gz
      t1c_brain_mask.nii.gz
      ...
```

## Temp-File Pattern

All steps write to a temporary file first, then replace the target in-place:
```python
temp_path = output_path.with_suffix('.tmp.nii.gz')
# ... write to temp_path ...
temp_path.rename(output_path)
```
This prevents partial writes from corrupting the dataset.

## Common Tasks

| Task | Where to look |
|------|---------------|
| Add a new preprocessing step | Create step handler in `steps/`, register in `preprocess.py`, add `StepMetadata` entry and config dataclass in `config.py` |
| Change step order | Modify `steps:` list in YAML config |
| Run a step twice with different params | Add e.g. `intensity_normalization_2` to `steps:` with its own `step_configs` entry |
| Debug a step failure | Check `{viz_root}/` for visualization PNGs, examine `{artifacts}/` for intermediate outputs |
| Process a single patient | Set `patient_selector: "single"` and `patient_id:` in config, or use `--patient` CLI flag |
