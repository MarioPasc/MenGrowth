# Phase 2: Completeness Filtering

## Theory

Completeness filtering removes studies and patients that lack sufficient data for longitudinal analysis. It operates in three stages:

1. **Orientation normalization:** When multiple files exist for the same modality but with different orientations (e.g., `t1c-axial.nrrd` and `t1c-sagital.nrrd`), the system selects one based on a priority order. The preferred sequence (exact match first, then axial, sagittal, coronal) is renamed to the canonical name (`t1c.nrrd`).

2. **Study-level filtering:** Each study is checked for the required modality set (default: t1c, t1n, t2w, t2f). Studies missing more than `allowed_missing_sequences_per_study` (default: 1) modalities are marked for deletion.

3. **Patient-level filtering:** Patients with fewer than `min_studies_per_patient` (default: 2) valid studies after study-level filtering are entirely removed. Longitudinal growth analysis requires at least two timepoints.

Optionally, non-required sequences can be removed to standardize the modality set across all studies.

## Motivation

Clinical datasets frequently have incomplete acquisitions — a patient may have T1c and FLAIR but no T2w for one study. Including severely incomplete studies degrades model training (requiring complex missing-data handling) and wastes downstream compute. The two-timepoint minimum ensures every retained patient contributes to longitudinal analysis.

## Code Map

- **Entry function:** `mengrowth/preprocessing/utils/filter_raw_data.py` → `filter_raw_data()`
- **Key subfunctions:**
  - `normalize_study_sequences()` — Orientation priority resolution
  - `scan_study_sequences()` — Glob `*.nrrd` and map to modality names
  - `filter_study()` — Check study completeness against requirements
  - `filter_patient()` — Check patient meets longitudinal minimum
  - `remove_non_required_sequences()` — Delete extra modalities
  - `apply_orientation_priority()` — Select best orientation variant
- **Config class:** `mengrowth/preprocessing/config.py` → `FilteringConfig`
- **YAML key:** `filtering` in `configs/raw_data.yaml`

## Config Reference

```yaml
filtering:
  sequences: [t1c, t1n, t2w, t2f]          # Required modality set
  allowed_missing_sequences_per_study: 1     # Max missing per study (1 = need 3 of 4)
  min_studies_per_patient: 2                 # Minimum timepoints for longitudinal
  orientation_priority: [none, axial, sagital, coronal]  # Preference order
  keep_only_required_sequences: false        # Remove extra modalities
  reid_patients: false                       # Deferred to Phase 4
```

## Inputs / Outputs

- **Input:** `{output_root}/dataset/MenGrowth-2025/P{N}/{study_num}/*.nrrd`
- **Output:** Same directory (studies/patients deleted in-place)
- **Rejected files:** Appended to `{output_root}/quality/rejected_files.csv` (stage 1)
- **Statistics returned:** `patients_kept`, `patients_removed`, `studies_processed`, `sequences_removed`

## Common Tasks

| Task | How |
|------|-----|
| Require all 4 modalities | Set `allowed_missing_sequences_per_study: 0` |
| Allow single-timepoint patients | Set `min_studies_per_patient: 1` |
| Add a new required modality | Append to `sequences` list |
| Debug a removed study | Check `rejected_files.csv` for stage=1 entries |
| Strip non-standard modalities | Set `keep_only_required_sequences: true` |
