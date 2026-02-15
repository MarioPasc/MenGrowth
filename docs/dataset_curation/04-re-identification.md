# Phase 4: Re-Identification and Anonymization

## Theory

Re-identification (reid) converts internal patient IDs (`P{N}`) to anonymized MenGrowth identifiers (`MenGrowth-XXXX`) and renames study directories to a consistent format (`MenGrowth-XXXX-YYY`). This is the final structural transformation before analysis.

The algorithm:
1. Collect all remaining `P*` directories and sort by numeric ID
2. Assign sequential MenGrowth IDs starting from 0001 (zero-padded to 4 digits)
3. For each patient, sort studies numerically and assign sequential study IDs (zero-padded to 3 digits, starting from 000)
4. Rename directories in-place
5. Write `id_mapping.json` recording the `P* → MenGrowth-*` mapping
6. Update clinical metadata with new IDs via `MetadataManager`

**Critical design decision:** Reid runs AFTER quality filtering. This ensures MenGrowth IDs are continuous with no gaps — if P5 was removed by quality filtering, the remaining patients get IDs MenGrowth-0001 through MenGrowth-XXXX without skipping any numbers.

## Motivation

- Raw `P{N}` IDs may contain identifying information or reveal enrollment order
- Continuous ID numbering is cleaner for publication and dataset distribution
- Standard `MenGrowth-XXXX-YYY` format enables consistent programmatic access
- The mapping file preserves traceability back to original IDs for debugging

## Code Map

- **Entry function:** `mengrowth/preprocessing/utils/filter_raw_data.py` → `reid_patients_and_studies()`
- **Metadata integration:** `mengrowth/preprocessing/utils/metadata.py` → `MetadataManager.set_mengrowth_id()`
- **ID normalization:** `MetadataManager._normalize_patient_id()` — handles bidirectional P* ↔ MenGrowth-* conversion
- **CLI flag:** `--skip-reid` (in `filter_raw_data()` via `skip_reid` parameter)

## Inputs / Outputs

- **Input:** `{output_root}/dataset/MenGrowth-2025/P{N}/{study_num}/*.nrrd`
- **Output:** `{output_root}/dataset/MenGrowth-2025/MenGrowth-{XXXX}/MenGrowth-{XXXX}-{YYY}/*.nrrd`
- **Mapping file:** `{output_root}/dataset/id_mapping.json`
- **Metadata update:** `MetadataManager` records `MenGrowth_ID` for each patient

### id_mapping.json Format

```json
{
  "P1": {
    "new_id": "MenGrowth-0001",
    "studies": {
      "0": "MenGrowth-0001-000",
      "1": "MenGrowth-0001-001"
    }
  },
  "P5": {
    "new_id": "MenGrowth-0002",
    "studies": {
      "0": "MenGrowth-0002-000"
    }
  }
}
```

## Common Tasks

| Task | How |
|------|-----|
| Skip re-identification | Use `--skip-reid` or set `skip_reid=True` |
| Trace original ID | Look up in `id_mapping.json` — keys are original P* IDs |
| Change ID format | Modify the formatting in `reid_patients_and_studies()` function |
| Verify ID continuity | Check that `id_mapping.json` has sequential MenGrowth-XXXX values with no gaps |
