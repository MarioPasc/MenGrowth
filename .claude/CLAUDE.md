# CLAUDE.md — MenGrowth

## Project Identity

**MenGrowth** is a fully automated, configuration-driven pipeline for longitudinal meningioma growth prediction research. It has two major stages: **Data Curation** transforms heterogeneous raw clinical MRI into a standardized, quality-controlled, anonymized NRRD dataset; **BraTS-like Preprocessing** transforms curated NRRD into analysis-ready NIfTI volumes suitable for deep learning.

- **Language:** Python >= 3.11
- **Domain:** Biomedical imaging / Neuro-oncology
- **Data formats:** NRRD (curation I/O), NIfTI (preprocessing output), XLSX (clinical metadata), YAML (config)
- **Core libraries:** SimpleITK, pynrrd, nibabel, NumPy, SciPy, pandas, matplotlib, seaborn, ANTs (nipype/antspyx)
- **Optional:** CuPy (GPU Sobel), HD-BET / SynthStrip (brain extraction), ECLARE (super-resolution), intensity-normalization (custom fork)

---

## Quick Reference

```bash
# Environment
~/.conda/envs/growth/bin/python

# Tests
~/.conda/envs/growth/bin/python -m pytest tests/ -v

# Data curation (full pipeline)
mengrowth-curate-dataset --config configs/raw_data.yaml --qa-config configs/templates/quality_analysis.yaml \
    --input-root /path/to/raw --output-root /path/to/curated --workers 4

# Preprocessing (full pipeline)
mengrowth-preprocess --config configs/icaiserver/preprocessing_icai.yaml [--patient MenGrowth-0015] [--verbose]

# Install
pip install -e .                   # Core
pip install -e ".[preprocessing]"  # With optional deps
```

---

## Two-Pipeline Architecture

```
Stage 1: Data Curation                         Stage 2: BraTS-like Preprocessing
═══════════════════════                         ════════════════════════════════
Raw NRRD (heterogeneous)                        Curated NRRD
  → [1. Reorganize]                               → [1. Data Harmonization]     (NRRD→NIfTI, RAS, bg zeroing)
  → [2. Completeness Filter]                      → [2. Bias Field Correction]  (N4ITK)
  → [3. Quality Filter]    (15 checks)            → [3. Resampling]             (BSpline/ECLARE/Composite)
  → [4. Re-ID/Anonymize]                          → [4. Cubic Padding]          (FOV normalization)
  → [5. Quality Analysis]                         → [5. Registration]           (coregistration + atlas)
  → [6. Visualization]                            → [6. Skull Stripping]        (HD-BET/SynthStrip)
  ↓                                               → [7. Intensity Normalization](z-score/KDE/FCM/...)
Curated NRRD Dataset                              → [8. Longitudinal Registration]
                                                  ↓
                                                Analysis-Ready NIfTI Dataset
```

---

## Data Curation Pipeline

**Entry:** `mengrowth/cli/curate_dataset.py` → `mengrowth-curate-dataset`
**Config:** `configs/raw_data.yaml` + `configs/templates/quality_analysis.yaml`
**Config parser:** `mengrowth/preprocessing/config.py` → `load_preprocessing_config()`

| Phase | Name | Doc | Code |
|-------|------|-----|------|
| 1 | Reorganize | [docs/dataset_curation/01-reorganization.md](../docs/dataset_curation/01-reorganization.md) | `preprocessing/utils/reorganize_raw_data.py` |
| 2 | Completeness Filter | [docs/dataset_curation/02-completeness-filtering.md](../docs/dataset_curation/02-completeness-filtering.md) | `preprocessing/utils/filter_raw_data.py` |
| 3 | Quality Filter | [docs/dataset_curation/03-quality-filtering.md](../docs/dataset_curation/03-quality-filtering.md) | `preprocessing/utils/quality_filtering.py` |
| 4 | Re-Identification | [docs/dataset_curation/04-re-identification.md](../docs/dataset_curation/04-re-identification.md) | `preprocessing/utils/filter_raw_data.py` |
| 5 | Quality Analysis | [docs/dataset_curation/05-quality-analysis.md](../docs/dataset_curation/05-quality-analysis.md) | `preprocessing/quality_analysis/analyzer.py` |
| 6 | Visualization | [docs/dataset_curation/06-visualization.md](../docs/dataset_curation/06-visualization.md) | `preprocessing/quality_analysis/visualize.py` |

Full technical reference: [docs/dataset_curation/dataset_curation_technical_reference.md](../docs/dataset_curation/dataset_curation_technical_reference.md)

---

## Preprocessing Pipeline

**Entry:** `mengrowth/cli/preprocess.py` → `mengrowth-preprocess`
**Orchestrator:** `mengrowth/preprocessing/src/preprocess.py` → `PreprocessingOrchestrator`
**Config:** `configs/icaiserver/preprocessing_icai.yaml`
**Config parser:** `mengrowth/preprocessing/src/config.py` → `load_preprocessing_pipeline_config()`

| Step | Name | Level | Doc | Code |
|------|------|-------|-----|------|
| 1 | Data Harmonization | modality | [docs/preprocessing/01-data-harmonization.md](../docs/preprocessing/01-data-harmonization.md) | `src/steps/data_harmonization.py` |
| 2 | Bias Field Correction | modality | [docs/preprocessing/02-bias-field-correction.md](../docs/preprocessing/02-bias-field-correction.md) | `src/steps/bias_field_correction.py` |
| 3 | Resampling | modality | [docs/preprocessing/03-resampling.md](../docs/preprocessing/03-resampling.md) | `src/steps/resampling.py` |
| 4 | Cubic Padding | study | [docs/preprocessing/04-cubic-padding.md](../docs/preprocessing/04-cubic-padding.md) | `src/steps/cubic_padding.py` |
| 5 | Registration | study | [docs/preprocessing/05-registration.md](../docs/preprocessing/05-registration.md) | `src/steps/registration.py` |
| 6 | Skull Stripping | study | [docs/preprocessing/06-skull-stripping.md](../docs/preprocessing/06-skull-stripping.md) | `src/steps/skull_stripping.py` |
| 7 | Intensity Normalization | modality | [docs/preprocessing/07-intensity-normalization.md](../docs/preprocessing/07-intensity-normalization.md) | `src/steps/intensity_normalization.py` |
| 8 | Longitudinal Registration | patient | [docs/preprocessing/08-longitudinal-registration.md](../docs/preprocessing/08-longitudinal-registration.md) | `src/steps/longitudinal_registration.py` |

Pipeline overview: [docs/preprocessing/00-pipeline-overview.md](../docs/preprocessing/00-pipeline-overview.md)

### Preprocessing Architecture

Steps are defined in YAML (`steps:` list) and can be reordered, repeated (e.g., `intensity_normalization_2`), or omitted. The `StepRegistry` maps step name patterns to handler functions via substring matching. Each step declares an execution level:

| Level | Scope | Examples |
|-------|-------|---------|
| `modality` | Per (patient, study, modality) | Data harmonization, bias correction, resampling, intensity normalization |
| `study` | Per (patient, study) | Cubic padding, registration, skull stripping |
| `patient` | Per patient (all studies) | Longitudinal registration |

### Detailed Patient Archive (HDF5)

**Code:** `mengrowth/preprocessing/src/archiver.py` → `DetailedPatientArchiver`
**Config:** `detailed_archive:` section in `general_configuration` (enabled per patient_ids list)
**Output:** `{output_root}/detailed_patient/{patient_id}/{study_id}/archive.h5`

Saves per-step MRI snapshots, brain masks, and registration transforms to HDF5 for showcase patients. Archive failure never halts the pipeline. Hooks in `PreprocessingOrchestrator`: init, metadata, modality step done, study step done, finalize.

---

## Segmentation Pipeline

**Entry:** `mengrowth/cli/segment.py` → `mengrowth-segment`
**Config:** `configs/picasso/segmentation.yaml`
**Subcommands:** `prepare`, `postprocess`, `cleanup`, `run`, `attach-to-archive`

| Stage | What | Code |
|-------|------|------|
| prepare | Discover studies, validate shapes, create BraTS-format input dir | `segmentation/prepare.py` |
| inference | Singularity container (BraTS 2025 1st-place) | `slurm/segmentation/meningioma_seg_worker.sh` |
| postprocess | Remap BraTS outputs back to study dirs as `seg.nii.gz` | `segmentation/postprocess.py` |
| attach-to-archive | Attach `seg.nii.gz` to HDF5 detailed archive | `preprocessing/src/archiver.py` |

**Shape correction:** `prepare_brats_input()` pads/crops (no interpolation) to exact `(240, 240, 155)`. Uses `shutil.copy2()` (not symlinks — Singularity bind mounts can't follow external symlinks).

**SLURM launcher:** `slurm/segmentation/meningioma_seg.sh` (defaults to `configs/picasso/segmentation.yaml`, supports `--depends-on JOB_ID`)

---

## Standardized Identifiers

| Entity | Format | Example |
|--------|--------|---------|
| Patient (raw) | `P{N}` | `P1`, `P42` |
| Patient (anon) | `MenGrowth-{XXXX}` | `MenGrowth-0001` |
| Study (anon) | `MenGrowth-{XXXX}-{YYY}` | `MenGrowth-0001-000` |
| Modality | `t1c`, `t1n`, `t2w`, `t2f` | — |

### Modality Synonym Resolution

| Standard | Known Synonyms |
|----------|---------------|
| `t1c` | T1ce, T1-ce, T1post, T1-post, T1 |
| `t1n` | T1pre, T1-pre, T1SIN, T1sin |
| `t2w` | T2, t2, T2-weighted |
| `t2f` | FLAIR, flair, T2-FLAIR, T2flair |

---

## Output Directory Structure

```
{curation_output}/
  dataset/
    MenGrowth-2025/MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nrrd
    id_mapping.json, metadata_enriched.csv, metadata_clean.json
  quality/
    rejected_files.csv, quality_issues.csv, quality_metrics.json
    qc_analysis/ (CSVs, JSONs, figures/*.png, quality_analysis_report.html)

{preprocessing_output}/
  MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nii.gz
{preprocessing_viz}/
  MenGrowth-XXXX/MenGrowth-XXXX-YYY/step{N}_{name}_{modality}.png
{preprocessing_artifacts}/
  MenGrowth-XXXX/MenGrowth-XXXX-YYY/{modality}_brain_mask.nii.gz, {modality}_bias_field.nii.gz
```

---

## Key Design Principles

1. **Non-destructive:** Curation copies data (`shutil.copy2()`); preprocessing writes to temp files then replaces in-place
2. **Reproducible:** All thresholds in YAML configs with code defaults. Deterministic `sorted()` on all directory listings
3. **Traceable:** Rejected files logged with reason/stage to CSV. Quality metrics exported to JSON
4. **Quality-first ordering:** Quality filtering before anonymization → gap-free MenGrowth IDs
5. **Parallel by default:** `ProcessPoolExecutor` (CPU-bound), `ThreadPoolExecutor` (I/O-bound)
6. **Configurable step order:** Preprocessing steps defined in YAML list, matched by `StepRegistry` pattern
7. **Singularity-safe I/O:** Never use symlinks for files passed to Singularity containers — bind mounts cannot follow symlinks pointing outside the mounted volume. Use `shutil.copy2()` instead.

---

## Coding Conventions

- **Config:** Pure `@dataclass` trees. No mutable defaults. All picklable for multiprocessing
- **Typing:** Full type annotations. `ValidationResult` for quality checks, `BasePreprocessingStep` ABC for preprocessing
- **Logging:** Python `logging` module. `--verbose` enables DEBUG level
- **Error handling:** Per-file/per-future exception catching. Failures logged and aggregated, never crash pipeline
- **File ops:** `shutil.copy2()` (curation), temp-file-then-rename (preprocessing), `Path.mkdir(parents=True, exist_ok=True)`
- **Determinism:** `sorted()` on all directory listings and post-parallel result collection
- **Artifact naming:** `{modality}_brain_mask.nii.gz`, `{modality}_bias_field.nii.gz`
- **Visualization naming:** `step{N}_{step_name}_{modality}.png`

---

## Common Tasks

### Curation

| Task | Where |
|------|-------|
| Add quality check | `quality_filtering.py` → function + config `@dataclass` in `config.py` |
| Change threshold | `configs/raw_data.yaml` → `quality_filtering` section |
| Add modality synonym | `configs/raw_data.yaml` → `modality_synonyms` |
| Add source directory | `reorganize_raw_data.py` → add scanner function |
| Add plot | `visualize.py` → method + toggle in `PlotConfig` |
| Debug rejection | `rejected_files.csv` → filter by `stage` and `rejection_reason` |
| Re-run analysis only | `--skip-reorganize --skip-filter --skip-quality-filter` |

### Preprocessing

| Task | Where |
|------|-------|
| Add preprocessing step | Handler in `src/steps/`, register in `preprocess.py`, config `@dataclass` + `StepMetadata` in `src/config.py` |
| Change step order | Modify `steps:` list in preprocessing YAML |
| Run step twice | Add e.g. `intensity_normalization_2` to `steps:` with own `step_configs` entry |
| Change method variant | Modify `method:` in the step's config (e.g., `"hdbet"` → `"synthstrip"`) |
| Debug step failure | Check viz PNGs in `{viz_root}/`, artifacts in `{artifacts}/` |
| Process single patient | `--patient MenGrowth-0015` or `patient_selector: "single"` in config |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_shapes.py <dataset_root>` | Report NIfTI shapes across preprocessed dataset (header-only, fast) |
