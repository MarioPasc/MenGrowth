# Phase 3: Quality Filtering

## Theory

Quality filtering applies a battery of 15 automated checks to every NRRD file, study, and patient. Each check returns a `ValidationResult` with a pass/fail status and an action (`warn` or `block`). Blocking failures cause file, study, or patient removal; warnings are logged but do not trigger removal.

### Check Categories

**A. Data Validation (header-only, fast):**
- **A1 — NRRD header:** Validates 3D dimensionality and presence of `space` field
- **A2 — Scout detection:** Detects localizer scans (thin slices, small matrix)
- **A3 — Voxel spacing:** Flags extreme spacings (< 0.3 mm or > 3.0 mm) and anisotropy (> 20:1)

**B. Image Quality (requires data loading):**
- **B1 — SNR:** Corner-based SNR estimation with Rayleigh noise correction; modality-specific thresholds
- **B2 — Contrast:** Coefficient of variation and uniform fraction checks
- **B3 — Intensity outliers:** Detects NaN/Inf values and extreme max/p99 ratios
- **B4 — Motion artifact:** Gradient entropy via 3D Sobel filters (optionally GPU-accelerated with CuPy)
- **B5 — Ghosting:** Corner-to-foreground intensity ratio

**C. Geometric Validation:**
- **C1 — Affine matrix:** Determinant range check (0.01–100)
- **C2 — FOV consistency:** Within-study and between-study FOV ratio limits
- **C3 — Orientation consistency:** Consistent `space` field across modalities (study-level)
- **C4 — Brain coverage:** Minimum spatial extent (100 mm in all dimensions)

**D. Longitudinal Validation (patient-level):**
- **D1 — Temporal ordering:** Verifies study indices increase monotonically
- **D3 — Modality consistency:** Checks identical modality sets across studies (disabled by default)

**E. Preprocessing Requirements:**
- **E1 — Registration reference:** Verifies at least one reference modality exists (t1n > t1c > t2f > t2w priority)

### Execution Model

1. **Validation phase** (parallel `ProcessPoolExecutor`): Each patient is validated independently, producing `PatientValidationReport` containing per-file, study-level, and patient-level results
2. **Removal phase** (sequential): Patients with blocking issues are pruned. If enough clean studies remain (≥ `min_studies_per_patient`), only blocked studies are removed; otherwise the entire patient is deleted

## Motivation

Clinical MRI data contains corrupted files, scout localizers, motion-degraded acquisitions, and geometrically inconsistent sequences. Without automated quality filtering, these degrade all downstream processing — particularly registration (which diverges on corrupt data) and intensity normalization (which produces meaningless statistics on noisy inputs).

## Code Map

- **Entry function:** `mengrowth/preprocessing/utils/quality_filtering.py` → `run_quality_filtering()`
- **Per-patient validation:** `_validate_patient()` → calls `validate_file()` per NRRD
- **Individual check functions:** `validate_nrrd_header()`, `detect_scout_localizer()`, `check_voxel_spacing()`, `check_snr()`, `check_contrast()`, `check_intensity_outliers()`, `check_motion_artifact()`, `check_ghosting()`, `validate_affine()`, `check_fov_consistency()`, `check_orientation_consistency()`, `check_brain_coverage()`, `check_temporal_ordering()`, `check_modality_consistency()`, `check_registration_reference()`
- **SNR computation:** `compute_snr_corner()`, `compute_snr_background()`
- **Export functions:** `export_quality_issues()`, `export_quality_metrics()`
- **Data classes:** `ValidationResult`, `FileValidationReport`, `PatientValidationReport`, `QualityFilteringStats`
- **Config class:** `mengrowth/preprocessing/config.py` → `QualityFilteringConfig` (15 nested sub-configs)
- **YAML key:** `quality_filtering` in `configs/raw_data.yaml`

## Config Reference

Key thresholds (all have code defaults):

```yaml
quality_filtering:
  enabled: true
  remove_blocked: true
  min_studies_per_patient: 2
  snr_filtering:
    modality_thresholds: {t1c: 8.0, t1n: 6.0, t2w: 5.0, t2f: 4.0}
    method: "corner"
    action: "block"
  contrast_detection:
    min_std_ratio: 0.10
    max_uniform_fraction: 0.95
    action: "block"
  motion_artifact:
    modality_thresholds: {t1c: 3.3, t1n: 3.0, t2w: 3.7, t2f: 2.7}
    action: "block"
  brain_coverage:
    min_extent_mm: 100.0
    action: "block"
```

## Inputs / Outputs

- **Input:** `{output_root}/dataset/MenGrowth-2025/P{N}/{study_num}/*.nrrd`
- **Output:** Same directory (blocked studies/patients deleted)
- **Exports:**
  - `{output_root}/quality/quality_issues.csv` — All failed checks with details
  - `{output_root}/quality/quality_metrics.json` — All raw metric values per file
  - `{output_root}/quality/rejected_files.csv` — Appended (stage 2)
- **Parallelization:** `ProcessPoolExecutor`, `CPU_COUNT // 2` workers

## Common Tasks

| Task | How |
|------|-----|
| Add a quality check | Add function in `quality_filtering.py` returning `ValidationResult`; add config `@dataclass` in `config.py`; wire into `validate_file()` or study/patient level |
| Change a threshold | Modify the relevant config in `configs/raw_data.yaml` under `quality_filtering` |
| Disable a check | Set `enabled: false` in that check's config section |
| Change warn to block | Set `action: "block"` in the check's config |
| Debug a rejection | Cross-reference `rejected_files.csv` (stage=2) with `quality_issues.csv` by `check_name` |
| Use GPU for motion check | Set `--workers 1` (CUDA fork safety); CuPy must be installed |
