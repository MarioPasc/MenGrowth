# Phase 1: Data Reorganization

## Theory

Raw clinical MRI data arrives in heterogeneous directory structures that reflect the originating institution's file management rather than any research-friendly convention. Data reorganization creates a standardized layout where each patient has a top-level directory containing numbered study subdirectories, each with modality-specific NRRD files using canonical names.

The reorganization process:
1. **Scans** multiple input source directories (baseline, controls, extensions) using source-specific scanner functions that understand each directory's naming convention
2. **Extracts** patient IDs from diverse formats (P1, 042, patient_01) and normalizes to `P{N}` format
3. **Resolves modality names** using case-insensitive substring matching against a configurable synonym table (e.g., "T1ce", "T1-post" → `t1c`)
4. **Filters** out unwanted files using exclusion glob patterns (segmentation masks, DICOM files, transform files)
5. **Copies** files to the standardized structure using `shutil.copy2()` for metadata preservation

All copies use a two-phase approach: first plan all operations sequentially (resolving modality names, detecting duplicates), then execute copies in parallel via `ThreadPoolExecutor`.

## Motivation

Multi-institutional meningioma datasets have scanner-specific directory structures, inconsistent modality naming (T1ce vs T1post vs T1-contrast), and mixed file types. Manual reorganization is error-prone and non-reproducible. Automated reorganization enables all downstream phases to assume a single, predictable directory structure.

## Code Map

- **Entry function:** `mengrowth/preprocessing/utils/reorganize_raw_data.py` → `reorganize_raw_data()`
- **Scanner functions:**
  - `scan_source_baseline()` — Baseline RM/TC directories → study 0
  - `scan_source_controls()` — control1, control2, ... → study 1, 2, ...
  - `scan_extension()` — primera, segunda, ... → study 0, 1, ...
- **Utilities:**
  - `extract_patient_id()` — Normalizes varied formats to `P{N}`
  - `should_exclude_file()` — Glob pattern matching against exclusion list
  - `copy_and_organize()` — Two-phase plan+execute with parallel copy
- **Config class:** `mengrowth/preprocessing/config.py` → `RawDataConfig`
- **YAML key:** `raw_data` in `configs/raw_data.yaml`

## Config Reference

```yaml
raw_data:
  study_mappings:
    baseline: "0"
    primera: "0"
    control1: "1"
    control2: "2"
    # ...
  modality_synonyms:
    t1c: [T1ce, T1-ce, T1post, T1-post, T1]
    t1n: [T1pre, T1-pre, T1SIN, T1sin]
    t2w: [T2, t2, T2-weighted]
    t2f: [FLAIR, flair, T2-FLAIR, T2flair]
  exclusion_patterns: ["*seg*", "*.dcm", "*transform*"]
  input_sources: [baseline, controls, extension_1]
```

## Inputs / Outputs

- **Input:** Heterogeneous raw directories (multiple source formats)
- **Output:** `{output_root}/dataset/MenGrowth-2025/P{N}/{study_num}/*.nrrd`
- **Rejected files:** Appended to `{output_root}/quality/rejected_files.csv` (stage 0)
- **Parallelization:** `ThreadPoolExecutor` for file copy, `CPU_COUNT // 2` workers

## Common Tasks

| Task | How |
|------|-----|
| Add a new source directory | Add scanner function in `reorganize_raw_data.py`, call from `reorganize_raw_data()` |
| Add modality synonym | Add to `modality_synonyms` in YAML config |
| Exclude a file pattern | Add glob pattern to `exclusion_patterns` |
| Debug a missing file | Check `rejected_files.csv` for stage=0 entries; verify modality synonym matching |
| Dry run | Pass `--dry-run` flag — logs all operations without copying |
