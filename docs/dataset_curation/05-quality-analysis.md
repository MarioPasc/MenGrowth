# Phase 5: Quality Analysis

## Theory

Quality analysis computes population-level metrics across the curated dataset, providing quantitative summaries of image characteristics and identifying statistical outliers. Unlike quality filtering (Phase 3) which gates individual files, quality analysis characterizes the dataset as a whole for reporting and downstream decisions.

The analysis pipeline:
1. **Scan dataset:** Enumerate all patients, studies, and sequences
2. **Compute per-image metrics:** Using SimpleITK for medical image I/O, compute:
   - Image dimensions (width, height, depth in voxels)
   - Voxel spacing (x, y, z in mm)
   - Intensity statistics (min, max, mean, std, range, percentiles, non-zero stats)
   - SNR estimate (background-based signal-to-noise ratio)
   - Contrast ratio (high vs low intensity regions)
   - Histogram (binned intensity distribution)
3. **Aggregate statistics:** Per-sequence and per-patient summaries with mean, std, min, max, median
4. **Detect outliers:** Using IQR (default) or Z-score methods to flag unusual images

Analysis runs in parallel via `ProcessPoolExecutor` at patient granularity.

## Motivation

Population-level QC metrics reveal systematic issues (e.g., one institution's scans consistently have lower SNR), guide threshold tuning for quality filtering, and provide the quantitative evidence needed for thesis/publication dataset description sections.

## Code Map

- **Orchestrator:** `mengrowth/preprocessing/quality_analysis/analyzer.py` → `QualityAnalyzer`
  - `run_analysis()` — Full pipeline: scan → analyze → aggregate → save
  - `scan_dataset()` — Enumerate patient/study/sequence structure
  - `analyze_patient()` — Per-patient metric computation (parallelizable unit)
- **Metric functions:** `mengrowth/preprocessing/quality_analysis/metrics.py`
  - `load_image()` — SimpleITK image loading (auto-detect NRRD/NIfTI)
  - `compute_image_dimensions()`, `compute_voxel_spacing()`
  - `compute_intensity_statistics()` — Min/max/mean/std/percentiles
  - `compute_snr_estimate()` — Background-based SNR
  - `compute_contrast_ratio()` — Percentile-based contrast
  - `compute_histogram()` — Binned intensity distribution
  - `detect_outliers_iqr()`, `detect_outliers_zscore()` — Outlier detection
  - `compute_patient_statistics()`, `compute_missing_sequences()`
  - `compute_spacing_statistics()`, `compute_dimension_statistics()`
- **Config class:** `mengrowth/preprocessing/config.py` → `QualityAnalysisConfig`
- **YAML file:** `configs/templates/quality_analysis.yaml`

## Config Reference

```yaml
quality_analysis:
  input_dir: "{output_root}/dataset/MenGrowth-2025"
  output_dir: "{output_root}/quality/qc_analysis"
  file_format: "auto"              # "auto" | "nrrd" | "nifti"
  expected_sequences: [t1c, t1n, t2w, t2f]
  intensity_percentiles: [1, 5, 25, 50, 75, 95, 99]
  metrics:
    compute_dimensions: true
    compute_spacing: true
    compute_intensity: true
    compute_snr: true
    compute_contrast: true
    compute_histogram: true
    compute_missing: true
  outlier_detection:
    method: "iqr"                  # "iqr" | "zscore"
    iqr_multiplier: 1.5
    zscore_threshold: 3.0
  parallel:
    enabled: true
    n_workers: 8
```

## Inputs / Outputs

- **Input:** `{output_root}/dataset/MenGrowth-2025/MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nrrd`
- **Outputs (all in `{output_root}/quality/qc_analysis/`):**
  - `per_study_metrics.csv` — One row per (study, sequence) with all metrics
  - `per_patient_summary.csv` — Aggregated per-patient statistics
  - `per_sequence_statistics.json` — Per-modality population statistics
  - `summary_statistics.json` — Overall dataset summary
  - `analysis_metadata.json` — Run metadata (config, timing, counts)
- **Parallelization:** `ProcessPoolExecutor`, default 8 workers

## Common Tasks

| Task | How |
|------|-----|
| Disable a metric | Set `compute_X: false` in `metrics` config |
| Change outlier sensitivity | Adjust `iqr_multiplier` (higher = fewer outliers) |
| Switch to Z-score outliers | Set `method: "zscore"` |
| Add a new metric | Add function in `metrics.py`; wire into `analyze_patient()` in `analyzer.py`; add to CSV schema |
| Re-run analysis only | `mengrowth-curate-dataset --skip-reorganize --skip-filter --skip-quality-filter ...` |
