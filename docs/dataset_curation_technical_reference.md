# MenGrowth Dataset Curation Pipeline: Technical Reference

> **Purpose:** This document provides a complete, scientifically rigorous description of the automated dataset curation pipeline used to transform heterogeneous, multi-institutional raw MRI data into a standardized, quality-controlled longitudinal dataset suitable for meningioma growth prediction research. It is intended as a primary source for thesis chapter writing.

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Software Architecture](#2-software-architecture)
3. [Input Data Specification](#3-input-data-specification)
4. [Phase 1: Data Reorganization](#4-phase-1-data-reorganization)
5. [Phase 2: Completeness Filtering](#5-phase-2-completeness-filtering)
6. [Phase 3: Quality Filtering](#6-phase-3-quality-filtering)
7. [Phase 4: Re-Identification and Anonymization](#7-phase-4-re-identification-and-anonymization)
8. [Phase 5: Quality Analysis](#8-phase-5-quality-analysis)
9. [Phase 6: Visualization and Reporting](#9-phase-6-visualization-and-reporting)
10. [Clinical Metadata Processing](#10-clinical-metadata-processing)
11. [Parallel Processing and Performance](#11-parallel-processing-and-performance)
12. [Output Specification](#12-output-specification)
13. [Configuration System](#13-configuration-system)
14. [Dependencies and Reproducibility](#14-dependencies-and-reproducibility)

---

## 1. Overview and Motivation

### 1.1 Problem Statement

Longitudinal meningioma growth studies require multimodal MRI data acquired across multiple timepoints per patient. In practice, raw clinical data arrives in heterogeneous directory structures with inconsistent naming conventions, variable acquisition parameters, missing sequences, and varying quality levels. Manual curation at this scale is error-prone, non-reproducible, and prohibitively time-consuming.

### 1.2 Pipeline Design

The MenGrowth curation pipeline is a fully automated, configuration-driven system implemented in Python (>=3.11) that transforms raw data through six sequential phases:

```
Raw Data (heterogeneous) ──> [1. Reorganize] ──> [2. Filter] ──> [3. Quality Filter]
    ──> [4. Re-Identify] ──> [5. Analyze] ──> [6. Visualize] ──> Curated Dataset
```

Key design principles:

- **Non-destructive:** All operations copy data; original raw files are never modified (`shutil.copy2()` preserves metadata).
- **Reproducible:** All thresholds and parameters are externalized to YAML configuration files.
- **Traceable:** Every rejected file is logged with reason, stage, and provenance to CSV.
- **Quality-first:** Quality filtering precedes anonymization to ensure continuous ID numbering.
- **Parallel:** CPU-bound validation uses `ProcessPoolExecutor`; I/O-bound copying uses `ThreadPoolExecutor`.

### 1.3 Entry Point

The unified CLI command is:

```bash
mengrowth-curate-dataset \
    --config configs/raw_data.yaml \
    --qa-config configs/templates/quality_analysis.yaml \
    --input-root /path/to/raw \
    --output-root /path/to/curated \
    --workers 4 \
    [--dry-run] [--verbose]
```

Individual phases can be skipped via `--skip-reorganize`, `--skip-filter`, `--skip-quality-filter`, `--skip-analyze`, and `--skip-visualize` flags.

---

## 2. Software Architecture

### 2.1 Module Organization

```
mengrowth/
  cli/
    curate_dataset.py          # Unified pipeline orchestrator
  preprocessing/
    config.py                  # Configuration dataclasses and YAML parsing
    utils/
      reorganize_raw_data.py   # Phase 1: Data reorganization
      filter_raw_data.py       # Phase 2: Completeness filtering + Phase 4: Reid
      quality_filtering.py     # Phase 3: Quality validation
      metadata.py              # Clinical metadata processing
    quality_analysis/
      analyzer.py              # Phase 5: Metric computation orchestrator
      metrics.py               # Metric computation functions (SimpleITK-based)
      visualize.py             # Phase 6: Plot generation and HTML reporting
```

### 2.2 Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pynrrd | latest | NRRD file I/O for quality filtering |
| SimpleITK | latest | Medical image I/O for quality analysis |
| NumPy | latest | Numerical computation |
| SciPy | latest | Sobel gradients, Shannon entropy |
| pandas | latest | Tabular data handling and CSV export |
| matplotlib + seaborn | latest | Visualization |
| openpyxl | latest | Excel clinical metadata parsing |
| PyYAML | latest | Configuration file parsing |
| CuPy (optional) | cuda12x | GPU-accelerated Sobel computation |

### 2.3 Data Formats

- **Input:** NRRD files (`.nrrd`), clinical metadata (`.xlsx`)
- **Intermediate:** NRRD files in standardized directory structure
- **Output:** NRRD files, CSV reports, JSON metrics, PNG plots, HTML report

---

## 3. Input Data Specification

### 3.1 Raw Data Directory Structure

The pipeline accepts data from three distinct source directories reflecting the institutional acquisition workflow:

```
{input_root}/
  source/
    baseline/
      RM/                          # Resonancia Magnetica (MRI)
        {modality}/                # e.g., FLAIR/, T1ce/, T1pre/, T2/
          {patient_id}/            # e.g., P1/, P42/
            {modality}_{patient}.nrrd
      TC/                          # Tomografia Computarizada (CT/contrast)
        {patient_id}/
          {filename}.nrrd
    controls/                      # Follow-up studies
      {patient_id}/
        control1/                  # 1st follow-up
          {modality_files}.nrrd
        control2/                  # 2nd follow-up
          {modality_files}.nrrd
        ...
  extension_1/                     # Extension cohort
    {numeric_id}/                  # e.g., 85/, 42/
      primera/                     # Baseline (Spanish ordinal)
        {modality_files}.nrrd
      segunda/                     # 1st follow-up
        {modality_files}.nrrd
      ...
```

### 3.2 Modality Naming Conventions

Raw filenames are highly variable. The pipeline maps them to four standardized modality identifiers:

| Standard Name | Description | Known Synonyms |
|---------------|-------------|----------------|
| `t1c` | T1-weighted contrast-enhanced | T1ce, T1-ce, T1post, T1-post, T1 |
| `t1n` | T1-weighted native (pre-contrast) | T1pre, T1-pre, T1SIN, T1sin |
| `t2w` | T2-weighted | T2, t2, T2-weighted |
| `t2f` | T2-weighted FLAIR | FLAIR, flair, T2-FLAIR, T2flair |

Additional modalities are recognized but not required: `swi` (SWI, SUSC), `dwi` (DWI, DIFUSION), `adc` (ADC), `ct` (TC, TOMOGRAFIA).

Orientation-suffixed variants are also handled: `t1c-axial`, `t1n-sagital`, `t1c-coronal`, etc.

### 3.3 Study Temporal Mappings

Study directories are mapped to numeric temporal indices via the configuration:

| Directory Name | Study Index | Source |
|----------------|-------------|--------|
| `baseline`, `primera` | 0 | Baseline study |
| `control1`, `segunda` | 1 | First follow-up |
| `control2`, `tercera` | 2 | Second follow-up |
| ... | ... | ... |
| `control10`, `undecima` | 10 | Tenth follow-up |

Both Spanish ordinals (masculine/feminine variants: `primera`/`primero`) and English control names are supported.

---

## 4. Phase 1: Data Reorganization

**Module:** `mengrowth/preprocessing/utils/reorganize_raw_data.py`

### 4.1 Objective

Transform the heterogeneous raw directory structure into a standardized, uniform layout where each file is unambiguously identified by patient, study, and modality.

### 4.2 Output Structure

```
{output_root}/dataset/MenGrowth-2025/
  P{id}/
    {study_number}/
      t1c.nrrd
      t1n.nrrd
      t2f.nrrd
      t2w.nrrd
      [dwi.nrrd, swi.nrrd, ...]   # Additional modalities if present
```

### 4.3 Algorithm

The reorganization proceeds through three scanning phases and one execution phase:

#### 4.3.1 Source Scanning

Three scanner functions traverse the raw data:

1. **`scan_source_baseline()`**: Scans `source/baseline/RM/` (modality-organized) and `source/baseline/TC/` (patient-organized). All files are assigned study number `"0"`.

2. **`scan_source_controls()`**: Scans `source/controls/{patient}/{controlN}/`. Maps control directory names to study numbers via `config.get_study_number()`.

3. **`scan_extension()`**: Scans `extension_1/{numeric_id}/{study_name}/`. Converts numeric patient IDs to P-prefix format (e.g., `85` -> `P85`).

Each scanner produces a list of `(source_path, patient_id, study_number)` tuples and populates a shared rejection list.

#### 4.3.2 Patient ID Normalization

The `extract_patient_id()` function ensures all IDs follow the `P{N}` format:
- `"P1"` -> `"P1"` (pass-through)
- `"85"` -> `"P85"` (numeric input)
- `"P042"` -> `"P42"` (leading zeros stripped)

#### 4.3.3 File Exclusion

Before inclusion, every file is tested against configurable glob patterns:

```yaml
exclusion_patterns:
  - "*seg*.nrrd"       # Segmentation masks
  - "*_mask.nrrd"      # Binary masks
  - "*_label.nrrd"     # Label maps
  - "*.h5"             # Transform files
  - "*.dcm"            # Raw DICOM files
  - "*Survey*.nrrd"    # Localizer/survey acquisitions
  - "*COLSAG*.nrrd"    # Localizer files
  - "*disc*"           # Hand-selected discards
```

#### 4.3.4 Modality Standardization

The `RawDataConfig.standardize_modality()` method performs case-insensitive substring matching against the configured synonym lists. For a file `FLAIR_P1.nrrd`:
1. Extract stem: `FLAIR_P1`
2. For each standard name and its synonyms, check if synonym is a case-insensitive substring of the stem
3. First match wins: `"FLAIR"` matches synonym of `t2f` -> returns `("t2f", True)`

Unmatched files are kept with their original stem and flagged in the rejection log.

#### 4.3.5 Copy Execution (Parallel)

The `copy_and_organize()` function operates in two phases:

**Phase 1 - Planning (sequential):**
- Resolve modality names for all files
- Detect duplicates via `(patient_id, study_number, standardized_filename)` composite key
- First occurrence wins; subsequent duplicates are rejected and logged
- Build a list of `(source, destination)` copy tasks

**Phase 2 - Execution (parallel):**
- Uses `ThreadPoolExecutor` with configurable worker count (default: `CPU_COUNT // 2`)
- Each task calls `shutil.copy2()` which preserves file metadata (timestamps, permissions)
- `mkdir(parents=True, exist_ok=True)` is thread-safe for directory creation
- Failed copies are caught per-future and added to the rejection log

### 4.4 Outputs

| File | Description |
|------|-------------|
| `{output}/dataset/MenGrowth-2025/` | Reorganized data directory tree |
| `{output}/dataset/rejected_files.csv` | Stage-0 rejection log |

### 4.5 Statistics Reported

```python
{"copied": int, "skipped": int, "errors": int}
```

---

## 5. Phase 2: Completeness Filtering

**Module:** `mengrowth/preprocessing/utils/filter_raw_data.py`

### 5.1 Objective

Ensure every retained study contains sufficient MRI sequences for downstream processing (registration, segmentation, analysis), and every retained patient has sufficient longitudinal coverage.

### 5.2 Filtering Criteria

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `sequences` | `[t1c, t1n, t2f, t2w]` | Required modalities per study |
| `allowed_missing_sequences_per_study` | `1` | Maximum missing sequences before study rejection |
| `min_studies_per_patient` | `2` | Minimum longitudinal timepoints per patient |
| `orientation_priority` | `[none, axial, sagital, coronal]` | Preference order for orientation-suffixed sequences |
| `keep_only_required_sequences` | `true` | Remove non-required modalities (DWI, SWI, etc.) |

### 5.3 Algorithm

#### 5.3.1 Sequence Normalization

For each study, the `normalize_study_sequences()` function resolves orientation-suffixed files to their canonical names:

1. Check if exact name exists (e.g., `t1c.nrrd`)
2. If not, iterate through orientation priority list:
   - `"none"` -> exact match (`t1c`)
   - `"axial"` -> suffixed match (`t1c-axial`)
   - `"sagital"` -> suffixed match (`t1c-sagital`)
   - `"coronal"` -> suffixed match (`t1c-coronal`)
3. First match is renamed: `t1c-axial.nrrd` -> `t1c.nrrd`

#### 5.3.2 Study-Level Filtering

For each study directory:
1. Normalize sequences using orientation priority
2. Count how many of the 4 required sequences are present after normalization
3. Compute `missing_count = |required| - |present|`
4. If `missing_count > allowed_missing_sequences_per_study` (> 1): **reject study**
5. Rejected study directories are deleted from disk (`shutil.rmtree`)

#### 5.3.3 Patient-Level Filtering

For each patient:
1. Filter all studies individually
2. Count surviving valid studies
3. If `valid_study_count < min_studies_per_patient` (< 2): **reject entire patient**
4. Rejected patient directories are deleted from disk
5. Metadata manager records exclusion reason

#### 5.3.4 Non-Required Sequence Cleanup

When `keep_only_required_sequences=true`, the `remove_non_required_sequences()` function iterates all surviving studies and deletes any `.nrrd` files whose stem is not in `{t1c, t1n, t2f, t2w}`. This removes ancillary modalities (DWI, SWI, ADC, CT) that are not used in the growth prediction pipeline.

### 5.4 Outputs

| File | Description |
|------|-------------|
| Updated `MenGrowth-2025/` | Filtered directory tree (rejected studies/patients deleted) |
| `{quality}/rejected_files.csv` | Stage-1 rejections appended to existing CSV |

### 5.5 Statistics Reported

```python
{
    "patients_kept": int,
    "patients_removed": int,
    "studies_processed": int,
    "files_renamed": int,
    "sequences_removed": int
}
```

---

## 6. Phase 3: Quality Filtering

**Module:** `mengrowth/preprocessing/utils/quality_filtering.py`

### 6.1 Objective

Validate the data integrity and image quality of every surviving NRRD file through a comprehensive battery of automated checks spanning five categories: data validation, image quality assessment, geometric validation, longitudinal consistency, and preprocessing readiness.

### 6.2 Validation Hierarchy

Quality checks are organized at three levels:

```
Patient Validation Report
  |-- Patient-level checks (D1, D3)
  |-- Study Validation Reports (per study)
        |-- Study-level checks (C3, E1)
        |-- File Validation Reports (per NRRD file)
              |-- File-level checks (A1-A3, B1-B5, C1-C2, C4)
```

Each check produces a `ValidationResult` with:
- `passed: bool` -- whether the check succeeded
- `check_name: str` -- identifier (e.g., `"snr_filtering"`)
- `message: str` -- human-readable explanation
- `action: str` -- `"warn"` (advisory) or `"block"` (triggers removal)
- `details: Dict` -- raw metric values for downstream analysis

### 6.3 Category A: Data Validation (Header-Only)

These checks examine NRRD header metadata without loading voxel data.

#### A1: NRRD Header Validation (`validate_nrrd_header`)

| Check | Threshold | Action |
|-------|-----------|--------|
| Dimensionality | `dimension == 3` | Block |
| Spatial metadata | `space` or `space directions` field must exist | Block |

**Rationale:** 2D slices and 4D time-series are incompatible with the 3D registration pipeline. Orientation metadata is required for spatial alignment.

#### A2: Scout/Localizer Detection (`detect_scout_localizer`)

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum dimension | `min(sizes[:3]) >= 10` voxels | Block |
| Maximum slice thickness | `max(spacings) <= 8.0` mm | Block |

**Rationale:** Scout/localizer images are low-resolution planning sequences with very few slices or extremely thick slices. They are not suitable for volumetric analysis.

**Algorithm:**
1. Read `sizes` from header (voxel counts per dimension)
2. Compute physical spacings from `space directions` via L2 norm of each direction vector
3. Reject if any dimension has fewer than 10 voxels or if maximum spacing exceeds 8.0 mm

#### A3: Voxel Spacing Validation (`check_voxel_spacing`)

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum spacing | `min(spacings) >= 0.3` mm | Warn |
| Maximum spacing | `max(spacings) <= 7.0` mm | Warn |
| Anisotropy ratio | `max/min <= 20.0` | Warn |

**Rationale:** Unreasonably small spacings suggest header corruption. Excessively large spacings indicate inadequate spatial resolution. High anisotropy is expected in 2D multi-slice FLAIR acquisitions (e.g., 0.4 mm in-plane, 5-6.5 mm slice thickness) and is therefore only warned, not blocked.

### 6.4 Category B: Image Quality Assessment (Requires Voxel Data)

These checks load the full 3D voxel array and compute signal-level metrics.

#### B1: Signal-to-Noise Ratio (`check_snr`)

**Method: Corner-based noise estimation (default)**

```
1. Sample 8 corner cubes of size c^3 (c = 10 voxels by default):
   corners = data[:c,:c,:c], data[:c,:c,-c:], ..., data[-c:,-c:,-c:]

2. Concatenate corner voxels and compute raw noise standard deviation:
   sigma_raw = std(corner_values)

3. Apply Rayleigh correction for magnitude MR images:
   sigma = sigma_raw * sqrt(2/pi)     [= sigma_raw * 0.7979]

4. Estimate signal as 75th percentile of foreground voxels:
   foreground = data[data > percentile(data, 10)]
   signal = percentile(foreground, 75)

5. SNR = signal / sigma
```

**Alternative method:** Background percentile estimation (uses 10th percentile of positive voxels as background threshold, computes noise as std of background, signal as mean of foreground).

| Modality | Threshold | Action |
|----------|-----------|--------|
| t1c | SNR >= 8.0 | Block |
| t1n | SNR >= 6.0 | Block |
| t2w | SNR >= 5.0 | Block |
| t2f | SNR >= 4.0 | Block |
| (fallback) | SNR >= 5.0 | Block |

**Rationale:** Modality-specific thresholds account for inherent SNR differences. Contrast-enhanced T1 (t1c) has the highest expected SNR; FLAIR (t2f) is noisier due to the inversion recovery pulse. Corner-based estimation is preferred over background-percentile because it avoids assumptions about the intensity distribution of the signal region.

#### B2: Contrast Detection (`check_contrast`)

| Check | Threshold | Action |
|-------|-----------|--------|
| All-zero image | `mean == 0` | Block |
| Low contrast | `std / |mean| < 0.10` | Block |
| Uniform image | `max(value_counts) / total_voxels > 0.95` | Block |

**Algorithm:**
1. Compute global mean and standard deviation
2. Reject if mean is zero (empty image)
3. Compute coefficient of variation: `cv = std / |mean|`
4. Compute uniform fraction: find most frequent voxel value, compute its fraction of total voxels
5. Reject if either threshold is violated

**Rationale:** Uniform or near-uniform images indicate acquisition failures, corrupted files, or zero-padded volumes.

#### B3: Intensity Outlier Detection (`check_intensity_outliers`)

| Check | Threshold | Action |
|-------|-----------|--------|
| NaN/Inf values | Any NaN or Inf present | Block |
| Extreme outliers | `max / percentile(99)` exceeds modality threshold | Warn |

Per-modality outlier ratio thresholds:

| Modality | Max/P99 Threshold |
|----------|-------------------|
| t1c | 10.0 |
| t1n | 15.0 |
| t2w | 12.0 |
| t2f | 30.0 |

**Rationale:** NaN/Inf values cause numerical failures in all downstream processing. Extreme intensity outliers (e.g., `max/p99 > 10`) suggest reconstruction artifacts or partial Fourier artifacts. FLAIR allows a higher ratio (30.0) because cerebrospinal fluid suppression creates a bimodal intensity distribution with naturally high max/p99 ratios.

#### B4: Motion Artifact Detection (`check_motion_artifact`)

**Algorithm:**
1. Cast data to float64
2. Compute 3D Sobel gradient magnitude:
   ```
   grad_mag = sqrt(sobel(data, axis=0)^2 + sobel(data, axis=1)^2 + sobel(data, axis=2)^2)
   ```
3. Build histogram of non-zero gradient magnitudes (256 bins)
4. Compute Shannon entropy of normalized histogram:
   ```
   H = -sum(p_i * log2(p_i))  for p_i > 0
   ```

| Check | Threshold | Action |
|-------|-----------|--------|
| Gradient entropy | `H >= 3.0` bits | Warn |

**Rationale:** Sharp anatomical edges produce high-entropy gradient histograms (diverse gradient magnitudes). Motion-blurred images have homogeneous, low-magnitude gradients, yielding low entropy. The threshold of 3.0 bits represents the boundary between acceptable and suspect image quality.

**GPU Acceleration:** When CuPy is available and `n_workers <= 1`, the Sobel computation is offloaded to GPU via `cupyx.scipy.ndimage.sobel()`, providing significant speedup for large 3D volumes. GPU is disabled in multi-process mode to avoid CUDA fork safety issues.

#### B5: Ghosting Artifact Detection (`check_ghosting`)

**Algorithm:**
1. Sample 8 corner cubes (same as SNR): compute `corner_mean = mean(|corner_voxels|)`
2. Compute foreground mean: `foreground_mean = mean(voxels > percentile(10))`
3. Ratio: `R = corner_mean / foreground_mean`

| Check | Threshold | Action |
|-------|-----------|--------|
| Corner/foreground ratio | `R <= 0.15` | Warn |

**Rationale:** In artifact-free images, volume corners should contain only air (near-zero signal). Ghosting artifacts replicate signal into the phase-encoding direction, elevating corner intensities. A ratio above 0.15 suggests clinically relevant ghosting.

### 6.5 Category C: Geometric Validation

#### C1: Affine Matrix Validation (`validate_affine`)

**Algorithm:**
1. Extract `space directions` matrix from NRRD header (3x3 rotation/scaling matrix)
2. Compute determinant: `det = |M|` (represents signed voxel volume in mm^3)
3. Validate absence of NaN/Inf in matrix

| Check | Threshold | Action |
|-------|-----------|--------|
| Matrix validity | No NaN/Inf | Block |
| Determinant minimum | `|det| >= 0.01` | Block |
| Determinant maximum | `|det| <= 100.0` | Block |

**Rationale:** The determinant encodes the voxel volume. For high-resolution data (0.24 mm isotropic), `det ~ 0.014`. For low-resolution data (3 mm), `det ~ 27`. Values outside [0.01, 100] indicate corrupted or nonsensical spatial transforms.

#### C2: Field-of-View Consistency (`check_fov_consistency`)

**Algorithm:**
1. Compute FOV per dimension: `FOV_i = sizes_i * spacing_i` (mm)
2. Compute asymmetry ratio: `R = max(FOV) / min(FOV)`

| Check | Threshold | Action |
|-------|-----------|--------|
| Moderate asymmetry | `R > 3.0` | Warn |
| Extreme asymmetry | `R > 5.0` | Block |

**Rationale:** Highly asymmetric FOVs cause issues with padding algorithms that assume roughly cubic volumes.

#### C3: Orientation Consistency Within Study (`check_orientation_consistency`)

**Algorithm:**
1. For each modality in the study, read the `space` field from the NRRD header
2. Collect unique orientation strings (excluding "unknown" and error values)
3. Flag if more than one unique orientation is found

| Check | Threshold | Action |
|-------|-----------|--------|
| Consistent orientations | All modalities have same `space` | Warn |

**Rationale:** Registration between modalities within a study assumes consistent orientation space.

#### C4: Brain Coverage Validation (`check_brain_coverage`)

**Algorithm:**
1. Compute physical extent per dimension: `extent_i = sizes_i * spacing_i` (mm)
2. Check minimum extent across all dimensions

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum physical extent | `min(extents) >= 100.0` mm | Block |

**Rationale:** A typical adult brain measures approximately 140-180 mm in each dimension. A minimum extent of 100 mm ensures that acquisitions with partial brain coverage (e.g., targeted slab acquisitions) are excluded, as they would produce incomplete registrations.

### 6.6 Category D: Longitudinal Consistency

#### D1: Temporal Ordering (`check_temporal_ordering`)

**Algorithm:**
1. Parse study numbers from study directory names (e.g., `MenGrowth-0001-002` -> 2)
2. Check if parsed numbers are in ascending order

| Check | Threshold | Action |
|-------|-----------|--------|
| Study ordering | Numbers are monotonically increasing | Warn |

**Rationale:** Misordered studies would produce incorrect growth trajectories. This check validates the integrity of study numbering.

#### D3: Modality Consistency (`check_modality_consistency`)

**Algorithm:**
1. For each study, collect the set of available modalities
2. Check if all studies have identical modality sets

| Check | Threshold | Action |
|-------|-----------|--------|
| Consistent modalities | All timepoints have same set | Warn |
| **Default state** | **Disabled** | - |

**Rationale:** Disabled by default because it is a strict requirement. The completeness filter already ensures minimum sequence requirements; some variation across timepoints is acceptable.

### 6.7 Category E: Preprocessing Readiness

#### E1: Registration Reference Availability (`check_registration_reference`)

**Algorithm:**
1. Parse priority string: `"t1n > t1c > t2f > t2w"` -> `["t1n", "t1c", "t2f", "t2w"]`
2. Check if at least one modality from the priority list exists in the study

| Check | Threshold | Action |
|-------|-----------|--------|
| Reference available | At least one priority modality present | Block |

**Rationale:** The preprocessing pipeline requires a reference modality for inter-modal registration. The priority order reflects typical reference quality: T1 native (t1n) provides the best anatomical contrast for registration; FLAIR (t2f) and T2-weighted (t2w) are less ideal but acceptable fallbacks.

### 6.8 Execution: Two-Phase Quality Filtering

The main `run_quality_filtering()` function executes in two phases:

#### Phase 1: Parallel Validation

```python
n_workers = n_workers or max(1, (os.cpu_count() or 4) // 2)

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    future_to_patient = {
        executor.submit(_validate_patient, patient_dir, config, None): patient_dir.name
        for patient_dir in patient_dirs
    }
    for future in as_completed(future_to_patient):
        patient_report, local_stats = future.result()
        # aggregate stats...
```

Each patient is validated independently by `_validate_patient()`:
1. Iterate all study directories
2. For each study, iterate all NRRD files and call `validate_file()` (runs checks A1-A3, B1-B5, C1-C2, C4)
3. Run study-level checks (C3: orientation consistency, E1: registration reference)
4. Run patient-level checks (D1: temporal ordering, D3: modality consistency)
5. Return `(PatientValidationReport, stats_dict)` -- all components are picklable dataclasses/dicts

**Picklability:** All function arguments are safe for multiprocessing:
- `patient_dir: Path` -- built-in picklable type
- `config: QualityFilteringConfig` -- pure `@dataclass` tree with primitive fields
- `metadata_manager=None` -- trivially picklable (not used in validation)

Results are sorted by `patient_id` after collection to ensure deterministic output regardless of execution order.

#### Phase 2: Actionable Removal (Sequential)

When `config.remove_blocked=True` and not in dry-run mode:

For each patient with blocking issues:
1. Count clean vs. blocked studies
2. If `clean_count >= min_studies_per_patient` (>= 2):
   - Delete only blocked study directories (`shutil.rmtree`)
   - Patient survives with reduced study count
3. Else:
   - Delete entire patient directory
   - Mark patient as excluded in metadata with aggregated reason string
4. Log all deleted files to rejection CSV

### 6.9 Quality Filtering Summary Table

| ID | Check Name | Category | Input | Threshold | Default Action |
|----|------------|----------|-------|-----------|----------------|
| A1 | NRRD Validation | Data | Header | 3D, space field | Block |
| A2 | Scout Detection | Data | Header | >=10 vox, <=8mm | Block |
| A3 | Voxel Spacing | Data | Header | 0.3-7.0mm, aniso<=20 | Warn |
| B1 | SNR Filtering | Quality | Voxels | Modality-specific (4-8) | Block |
| B2 | Contrast Detection | Quality | Voxels | cv>=0.10, uniform<95% | Block |
| B3 | Intensity Outliers | Quality | Voxels | No NaN/Inf; max/p99 | Warn |
| B4 | Motion Artifact | Quality | Voxels | Entropy>=3.0 bits | Warn |
| B5 | Ghosting Detection | Quality | Voxels | Corner/fg<=0.15 | Warn |
| C1 | Affine Validation | Geometry | Header | 0.01<=|det|<=100 | Block |
| C2 | FOV Consistency | Geometry | Header | Ratio<=3(w)/5(b) | Warn/Block |
| C3 | Orientation | Geometry | Headers | All same space | Warn |
| C4 | Brain Coverage | Geometry | Header | Extent>=100mm | Block |
| D1 | Temporal Order | Longitudinal | IDs | Monotonic increase | Warn |
| D3 | Modality Consistency | Longitudinal | Sets | Identical across time | Warn (disabled) |
| E1 | Registration Ref | Preprocessing | Set | t1n>t1c>t2f>t2w | Block |

### 6.10 Outputs

| File | Description |
|------|-------------|
| Updated `MenGrowth-2025/` | Blocked studies/patients removed |
| `{quality}/rejected_files.csv` | Stage-2 rejections appended |
| `{quality}/quality_issues.csv` | All failed checks (CSV format) |
| `{quality}/quality_metrics.json` | All computed metrics per file (JSON, for visualization) |

---

## 7. Phase 4: Re-Identification and Anonymization

**Module:** `mengrowth/preprocessing/utils/filter_raw_data.py` (`reid_patients_and_studies()`)

### 7.1 Objective

Replace patient-identifiable directory names with anonymous, continuous identifiers. By running reid after quality filtering, the resulting IDs have no gaps.

### 7.2 Algorithm

```python
# Sort patients by numeric part: P1, P2, P5, P42 -> sorted order
patient_dirs = sorted(
    [d for d in mengrowth_dir.iterdir() if d.is_dir() and d.name.startswith("P")],
    key=lambda p: int(p.name[1:])
)

for counter, patient_dir in enumerate(patient_dirs, start=1):
    new_patient_id = f"MenGrowth-{counter:04d}"     # e.g., MenGrowth-0001

    for study_idx, study_dir in enumerate(sorted_studies):
        new_study_id = f"{new_patient_id}-{study_idx:03d}"  # e.g., MenGrowth-0001-000
        study_dir.rename(patient_dir / new_study_id)

    patient_dir.rename(mengrowth_dir / new_patient_id)
```

### 7.3 ID Format

- **Patient:** `MenGrowth-XXXX` (4-digit zero-padded, supports up to 9999 patients)
- **Study:** `MenGrowth-XXXX-YYY` (3-digit zero-padded, supports up to 1000 timepoints per patient)

### 7.4 ID Mapping

The complete mapping is persisted as `{output}/dataset/id_mapping.json`:

```json
{
    "P1": {
        "new_id": "MenGrowth-0001",
        "studies": {
            "0": "MenGrowth-0001-000",
            "1": "MenGrowth-0001-001"
        }
    },
    "P42": {
        "new_id": "MenGrowth-0002",
        "studies": {
            "0": "MenGrowth-0002-000",
            "2": "MenGrowth-0002-001"
        }
    }
}
```

Note: study `"2"` (not `"1"`) may map to `MenGrowth-0002-001` if study 1 was removed during filtering.

### 7.5 Metadata Integration

After physical renaming, the id mapping is applied to the `MetadataManager`:
- `set_mengrowth_id(original_id, new_id)` records the mapping
- Internal lookups support both `P1` and `MenGrowth-0001` formats

---

## 8. Phase 5: Quality Analysis

**Module:** `mengrowth/preprocessing/quality_analysis/analyzer.py` + `metrics.py`

### 8.1 Objective

Compute comprehensive spatial, intensity, and signal quality metrics across the entire curated dataset for population-level characterization and outlier identification.

### 8.2 Dataset Scanning

`QualityAnalyzer.scan_dataset()` traverses the curated directory tree and builds:

```python
{
    "MenGrowth-0001": {
        "MenGrowth-0001-000": ["t1c", "t1n", "t2w", "t2f"],
        "MenGrowth-0001-001": ["t1c", "t1n", "t2w", "t2f"],
    },
    ...
}
```

File format is auto-detected: `.nrrd`, `.nii.gz`, or `.nii`.

### 8.3 Per-Image Metrics

For each image file, the following metrics are computed using SimpleITK:

#### 8.3.1 Spatial Metrics

- **Voxel spacing** `(x, y, z)` in mm via `sitk.Image.GetSpacing()`
- **Image dimensions** `(width, height, depth)` in voxels via `sitk.Image.GetSize()`

#### 8.3.2 Intensity Statistics

Computed from the numpy array representation (`sitk.GetArrayFromImage()`):

| Metric | Description |
|--------|-------------|
| `min`, `max`, `mean`, `std` | Basic descriptive statistics |
| `range` | `max - min` |
| `p1, p5, p25, p50, p75, p95, p99` | Configurable percentiles |
| `non_zero_mean`, `non_zero_std` | Statistics excluding background (zero voxels) |
| `non_zero_fraction` | Fraction of voxels above zero |

#### 8.3.3 Signal Quality

- **SNR estimate:** Background-based method using 5th percentile as background threshold:
  ```
  signal = mean(voxels > 0)
  noise = std(voxels <= percentile(5))
  SNR = signal / noise
  ```
- **Contrast ratio:** Percentile-based Michelson contrast:
  ```
  CR = (mean(voxels >= p75) - mean(voxels <= p25)) / (mean(voxels >= p75) + mean(voxels <= p25))
  ```

### 8.4 Aggregation and Outlier Detection

#### 8.4.1 Per-Sequence Statistics

For each modality (t1c, t1n, t2w, t2f), the following are computed across all instances:

- **Spacing:** mean, std, min, max, median per axis
- **Dimensions:** mean, std, min, max, median per axis
- **Intensity:** mean and std of per-study means, mean of per-study stds
- **SNR:** mean, std, min, max

#### 8.4.2 Outlier Detection

Two methods are available (configured via `outlier_detection.method`):

**IQR method (default):**
```
Q1 = percentile(values, 25)
Q3 = percentile(values, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = values outside [lower_bound, upper_bound]
```

**Z-score method:**
```
outliers = values where |z| > 3.0
bounds = mean +/- 3.0 * std
```

#### 8.4.3 Missing Sequence Analysis

For each expected sequence:
- `present_count`: number of studies containing the sequence
- `missing_count`: number of studies lacking the sequence
- `missing_fraction`: `missing_count / total_studies`
- `overall_missing_rate`: total missing / total expected

### 8.5 Parallelization

Analysis is parallelized at the patient level using `ProcessPoolExecutor`:
- Each patient's studies are analyzed by `analyze_patient()` in a separate process
- Worker count is configurable (default: 8 in quality analysis config)
- Results are collected and aggregated in the main thread

### 8.6 Outputs

| File | Description |
|------|-------------|
| `per_study_metrics.csv` | One row per (study, sequence) with all metrics |
| `per_patient_summary.csv` | One row per patient with study counts |
| `per_sequence_statistics.json` | Aggregated per-modality statistics |
| `summary_statistics.json` | Dataset-wide summary |
| `analysis_metadata.json` | Timestamp, config, and provenance |

---

## 9. Phase 6: Visualization and Reporting

**Module:** `mengrowth/preprocessing/quality_analysis/visualize.py`

### 9.1 Objective

Generate publication-quality plots and a comprehensive HTML report summarizing dataset characteristics, quality filtering results, and clinical metadata.

### 9.2 Plot Generation

The `QualityVisualizer` generates six configurable plot types:

| Plot | Type | Description |
|------|------|-------------|
| Studies per Patient | Histogram | Distribution of longitudinal timepoint counts |
| Missing Sequences | Heatmap | Patient x Modality binary matrix showing data availability |
| Spacing Distribution | Violin | Per-modality voxel spacing distributions (x, y, z axes) |
| Intensity Statistics | Boxplot | Per-modality intensity mean/std distributions |
| Dimension Consistency | Scatter | Image dimensions per modality, identifies outliers |
| SNR Distribution | Distribution | Per-modality SNR estimates |

**Figure settings:**
- Resolution: 300 DPI
- Format: PNG (configurable to PDF or SVG)
- Default size: 10 x 6 inches
- Color palette: Seaborn "Set2"
- Style: `sns.set_style("whitegrid")`

### 9.3 Quality Filtering Visualizations

When quality metrics JSON is available, three additional plots are generated:

1. **Quality Filtering Summary**: Overall pass/warn/block counts for files, studies, and patients
2. **Issues by Type**: Bar chart of blocking issue frequencies per check type
3. **Quality Checks Heatmap**: Patient x Check Type binary matrix showing which checks failed

### 9.4 HTML Report

The `generate_html_report()` method produces a self-contained HTML document with:

1. **Summary Statistics Table:** Total patients, studies, sequences, missing data
2. **Clinical Metadata Section** (when metadata manager is provided):
   - Demographics (age distribution, sex distribution)
   - Tumor characteristics (initial volume, growth status)
3. **Quality Filtering Results:**
   - Pass/warn/block statistics
   - Issue type breakdown
   - Patient-level pass/fail status
4. **Quality Metric Plots:** All generated figures embedded as images
5. **Data Issues Table:** Detailed listing of all failed quality checks

---

## 10. Clinical Metadata Processing

**Module:** `mengrowth/preprocessing/utils/metadata.py`

### 10.1 Data Source

Clinical metadata is loaded from a structured Excel file (`.xlsx`) with multi-level column headers encoded in Spanish medical terminology. The file contains:

| Column Group | Fields |
|-------------|--------|
| `general` | Patient ID, age, sex (0=male, 1=female), medical history |
| `first_study/rm` | MRI date, equipment |
| `first_study/tc` | CT date, technique |
| `first_study/measurements` | Tumor dimensions (cc, ll, ap in mm), volume, calcification type |
| `first_study/attributes` | Lobulation, hyperostosis, edema, visual calcification scale, T2 signal, enhancement pattern, location |
| `c1` through `c5` | Follow-up studies with date, volume, calcification, edema |
| `groundtruth` | Progressive calcification indicator, growth indicator |

### 10.2 Processing Pipeline

1. **`apply_hardcoded_codification()`**: Reads Excel file and maps Spanish column headers to standardized English names using hardcoded dictionaries. Handles multi-level column hierarchy, non-breaking spaces, and case normalization.

2. **`create_json_from_csv()`**: Converts the codified CSV to a nested JSON structure. Empty control blocks (volume=0/None and all other fields None) are automatically pruned.

3. **`PatientMetadata.from_json_dict()`**: Constructs typed dataclass instances with parsed numeric values and safe integer conversion (handles comma-separated values like `"1, 5"`).

### 10.3 MetadataManager Class

The `MetadataManager` class provides a unified interface for metadata operations throughout the pipeline:

**Loading:**
- `load_from_xlsx(xlsx_path)`: Full processing pipeline (xlsx -> CSV -> JSON -> PatientMetadata)
- `load_from_json(json_path)`: Load from previously exported JSON
- `load_from_enriched_csv(csv_path)`: Load from previously exported enriched CSV

**Patient ID Resolution:**
- `_normalize_patient_id()`: Handles `P1`, `1`, `P042`, and `MenGrowth-0001` formats
- Internal bi-directional mapping: `_id_map` (P* -> MenGrowth-*) and `_reverse_id_map` (MenGrowth-* -> P*)

**Lifecycle Tracking:**
- `ensure_patient_exists(patient_id)`: Auto-creates stub entries for patients found in data but absent from metadata
- `mark_excluded(patient_id, reason)`: Records exclusion with reason string
- `mark_included(patient_id)`: Clears exclusion status
- `set_mengrowth_id(original_id, mengrowth_id)`: Records anonymized ID mapping

**Clinical Summary:**
- `get_clinical_summary()`: Returns aggregated statistics (age range/mean/median, sex distribution, growth status, volume statistics, exclusion reasons)
- `get_volume_progression_data()`: Returns time-series volume data for included patients (baseline + controls)

### 10.4 Metadata Outputs

| File | Format | Description |
|------|--------|-------------|
| `metadata_enriched.csv` | CSV | patient_id, age, sex, medical_history, first_study_volume, growth, num_controls, included, exclusion_reason, MenGrowth_ID |
| `metadata_clean.json` | JSON | Full clinical data per patient with `_curation` tracking block |

---

## 11. Parallel Processing and Performance

### 11.1 Parallelization Strategy

| Phase | Bottleneck | Executor | Parallelization Unit | Default Workers |
|-------|-----------|----------|---------------------|-----------------|
| Reorganization | I/O (file copy) | `ThreadPoolExecutor` | Single file copy | `CPU_COUNT // 2` |
| Quality Filtering | CPU (Sobel, percentiles) | `ProcessPoolExecutor` | Single patient | `CPU_COUNT // 2` |
| Quality Analysis | CPU (image loading, stats) | `ProcessPoolExecutor` | Single patient | 8 (configurable) |

### 11.2 Worker Count Selection

The default `CPU_COUNT // 2` was chosen because the pipeline is designed to run on a workstation where the user is concurrently working, leaving half the cores free for other tasks.

The `--workers` CLI flag allows override:
```bash
mengrowth-curate-dataset --workers 8    # Use 8 workers
mengrowth-curate-dataset --workers 1    # Sequential mode (enables GPU)
```

### 11.3 GPU Acceleration

When CuPy (CUDA) is available and `n_workers <= 1`, the Sobel gradient computation in `check_motion_artifact()` is offloaded to GPU:

```python
if HAS_CUPY and n_workers <= 1:
    data_gpu = cp.asarray(data)
    grad_mag_gpu = cp.zeros_like(data_gpu)
    for axis in range(3):
        grad_mag_gpu += gpu_sobel(data_gpu, axis=axis) ** 2
    grad_mag = cp.asnumpy(cp.sqrt(grad_mag_gpu))
```

GPU acceleration is incompatible with `ProcessPoolExecutor` due to CUDA context fork safety issues. In multi-worker mode, the CPU Sobel path is used, which is the larger win due to patient-level parallelism.

---

## 12. Output Specification

### 12.1 Complete Directory Tree

```
{output_root}/
  dataset/
    MenGrowth-2025/
      MenGrowth-0001/
        MenGrowth-0001-000/
          t1c.nrrd
          t1n.nrrd
          t2f.nrrd
          t2w.nrrd
        MenGrowth-0001-001/
          t1c.nrrd
          t1n.nrrd
          t2f.nrrd
          t2w.nrrd
      MenGrowth-0002/
        ...
    id_mapping.json
    metadata_enriched.csv
    metadata_clean.json
  quality/
    rejected_files.csv
    quality_issues.csv
    quality_metrics.json
    qc_analysis/
      per_study_metrics.csv
      per_patient_summary.csv
      per_sequence_statistics.json
      summary_statistics.json
      analysis_metadata.json
      figures/
        studies_per_patient.png
        missing_sequences.png
        spacing_violin.png
        intensity_boxplots.png
        dimension_scatter.png
        snr_distribution.png
        quality_filtering_summary.png
        quality_issues_by_type.png
        quality_checks_heatmap.png
      quality_analysis_report.html
```

### 12.2 File Schemas

#### rejected_files.csv

```
source_path | filename | patient_id | study_name | rejection_reason | source_type | stage
```

Where `stage`: 0 = reorganization, 1 = completeness filtering, 2 = quality filtering.

#### quality_issues.csv

```
patient_id | study_id | modality | file_path | check_name | action | message | level | details
```

Where `level`: "file", "study", or "patient".

#### quality_metrics.json

Hierarchical structure containing ALL computed metric values (both passing and failing) for every file in the dataset, enabling downstream reanalysis without recomputation.

#### id_mapping.json

Bidirectional mapping from original P* identifiers to anonymized MenGrowth-XXXX identifiers, including per-study mappings.

---

## 13. Configuration System

### 13.1 Architecture

All pipeline parameters are externalized to YAML configuration files and parsed into typed Python dataclasses via `config.py`. The system uses two configuration files:

1. **`raw_data.yaml`**: Controls reorganization, filtering, quality filtering, and metadata processing
2. **`quality_analysis.yaml`**: Controls quality analysis metrics and visualization

### 13.2 Configuration Hierarchy

```
PreprocessingConfig
  |-- RawDataConfig              # Reorganization settings
  |     |-- study_mappings       # Directory name -> study number
  |     |-- modality_synonyms    # Standardized name -> synonym list
  |     |-- exclusion_patterns   # File exclusion globs
  |     |-- output_structure     # Output path template
  |-- FilteringConfig            # Completeness filtering
  |     |-- sequences            # Required modality list
  |     |-- allowed_missing_sequences_per_study
  |     |-- min_studies_per_patient
  |     |-- orientation_priority
  |     |-- keep_only_required_sequences
  |     |-- reid_patients
  |-- MetadataConfig             # Clinical metadata
  |     |-- xlsx_path, enabled, output names
  |-- QualityFilteringConfig     # Quality validation
        |-- enabled, remove_blocked, min_studies_per_patient
        |-- NRRDValidationConfig       (A1)
        |-- ScoutDetectionConfig       (A2)
        |-- VoxelSpacingConfig         (A3)
        |-- SNRFilteringConfig         (B1)
        |-- ContrastDetectionConfig    (B2)
        |-- IntensityOutliersConfig    (B3)
        |-- MotionArtifactConfig       (B4)
        |-- GhostingDetectionConfig    (B5)
        |-- AffineValidationConfig     (C1)
        |-- FOVConsistencyConfig       (C2)
        |-- OrientationConsistencyConfig (C3)
        |-- BrainCoverageConfig        (C4)
        |-- TemporalOrderingConfig     (D1)
        |-- ModalityConsistencyConfig  (D3)
        |-- RegistrationReferenceConfig (E1)

QualityAnalysisConfig
  |-- MetricsConfig              # Which metrics to compute
  |-- OutlierDetectionConfig     # IQR or Z-score parameters
  |-- ParallelConfig             # Worker count
  |-- VisualizationConfig        # Plot generation
  |     |-- PlotConfig           # Individual plot toggles
  |     |-- FigureConfig         # DPI, format, size
  |     |-- HtmlReportConfig     # HTML report settings
  |-- OutputConfig               # Which files to save
```

### 13.3 Default Values Reference

Every threshold has a programmatic default in the dataclass definition, ensuring the pipeline runs with sensible defaults even when configuration entries are omitted. All defaults are documented in Section 6 (Quality Filtering) and in the configuration dataclass docstrings in `config.py`.

---

## 14. Dependencies and Reproducibility

### 14.1 Software Requirements

- **Python:** >= 3.11
- **Core libraries:** PyYAML, SimpleITK, NumPy, SciPy, pandas, matplotlib, seaborn, nibabel >= 5.0.0, nipype >= 1.8.6, pynrrd, scikit-image, openpyxl
- **Optional (preprocessing):** brats == 0.1.6 (brain extraction), intensity-normalization (custom fork at commit `6d2e486`), CuPy (CUDA 12.x, GPU-accelerated Sobel)

### 14.2 Installation

```bash
pip install -e .                  # Core installation
pip install -e ".[preprocessing]" # With optional preprocessing dependencies
```

### 14.3 Reproducibility Guarantees

1. **Deterministic output:** Patient reports are sorted by ID after parallel collection. File scanning uses `sorted()` on directory listings. Reid numbering follows numeric sort order.
2. **Configuration versioning:** All threshold values are in YAML files that can be version-controlled alongside the code.
3. **Provenance tracking:** `analysis_metadata.json` records timestamp, input/output paths, and configuration state. `rejected_files.csv` records every rejected file with its rejection stage and reason.
4. **Dry-run mode:** `--dry-run` simulates all operations without modifying any files, enabling pre-flight validation.

---

*Document generated from codebase analysis. All threshold values, algorithm descriptions, and data structures are derived directly from the source code as of 2026-02-07.*
