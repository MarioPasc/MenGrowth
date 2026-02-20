#!/usr/bin/env python3
"""Generate thesis data files from the MenGrowth curation pipeline outputs.

Produces 6 files:
  1. raw_cohort_summary.json        - Pre-filtering cohort statistics
  2. clinical_metadata_summary.json - Demographics and clinical characteristics
  3. pipeline_attrition.json        - Patient/study counts at each stage
  4. quality_check_results.csv      - Per-file quality check results
  5. spacing_summary.csv            - Per-file voxel spacing from curated dataset
  6. rejected_files_summary.csv     - Aggregated rejection counts

Usage:
    python scripts/generate_thesis_data.py
"""

import csv
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mengrowth.preprocessing.utils.metadata import MetadataManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_ROOT = Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/raw/processed")
CURATED_ROOT = Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated")
DATASET_DIR = CURATED_ROOT / "dataset" / "MenGrowth-2025"
QUALITY_DIR = CURATED_ROOT / "quality"
QC_DIR = QUALITY_DIR / "qc_analysis"
XLSX_PATH = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/raw/source/"
    "Meningioma_Adquisition/metadata.xlsx"
)
ID_MAPPING_PATH = CURATED_ROOT / "dataset" / "id_mapping.json"
METADATA_CSV_PATH = CURATED_ROOT / "dataset" / "metadata_enriched.csv"

OUTPUT_DIR = PROJECT_ROOT / "thesis_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Canonical modality set for the pipeline
REQUIRED_MODALITIES = {"t1c", "t1n", "t2w", "t2f"}

# Modality synonym map (from raw_data.yaml)
MODALITY_SYNONYMS = {
    "t2f": ["FLAIR", "flair", "Flair", "T2-FLAIR", "T2flair"],
    "t1n": ["T1pre", "T1-pre", "T1SIN", "T1sin"],
    "t1c": ["T1ce", "T1-ce", "T1post", "T1-post", "T1"],
    "t2w": ["T2", "t2", "T2-weighted"],
    "swi": ["SWI", "swi", "Swi", "SUSC"],
    "dwi": ["DWI", "dwi", "DIFUSION1", "DIFUSION2", "DIFUSION3", "Diffusion"],
    "adc": ["ADC", "adc"],
    "ct": ["TC", "Ct", "ct", "TOMOGRAFIA"],
}


def _resolve_modality(name: str) -> str:
    """Resolve a modality name to its canonical form."""
    name_lower = name.lower().strip()
    # Direct match
    if name_lower in MODALITY_SYNONYMS:
        return name_lower
    # Check all synonym lists
    for canonical, synonyms in MODALITY_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() == name_lower:
                return canonical
    # Check compound names with orientation suffixes
    for canonical in MODALITY_SYNONYMS:
        if name_lower.startswith(canonical):
            return canonical
    return name_lower


# ── 1. Raw Cohort Summary ──────────────────────────────────────────────────────


def generate_raw_cohort_summary() -> dict:
    """Scan raw data directories to produce pre-filtering cohort statistics."""
    logger.info("Generating raw_cohort_summary.json ...")

    # We reconstruct the Phase 1 (reorganized) state by combining:
    #   - Current curated dataset files (mapped back via id_mapping)
    #   - Quality-filtered rejected files (quality/rejected_files.csv, stages 1+2)
    # This gives us what Phase 1 output looked like before Phase 2 filtering.

    # Also scan the raw directory for the truly raw counts.
    raw_patients = set()
    raw_studies = 0
    raw_files = 0
    raw_modalities: Counter = Counter()

    # -- Scan source/baseline/RM (modality-first organization) --
    rm_dir = RAW_ROOT / "source" / "baseline" / "RM"
    if rm_dir.exists():
        for modality_dir in sorted(rm_dir.iterdir()):
            if not modality_dir.is_dir():
                continue
            mod_name = _resolve_modality(modality_dir.name)
            for patient_dir in sorted(modality_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                pid = patient_dir.name  # e.g., P1
                raw_patients.add(pid)
                nrrd_files = list(patient_dir.glob("*.nrrd"))
                raw_files += len(nrrd_files)
                if nrrd_files:
                    raw_modalities[mod_name] += 1  # count per patient

    # -- Scan source/baseline/TC --
    tc_dir = RAW_ROOT / "source" / "baseline" / "TC"
    if tc_dir.exists():
        for patient_dir in sorted(tc_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            raw_patients.add(patient_dir.name)
            nrrd_files = list(patient_dir.glob("*.nrrd"))
            raw_files += len(nrrd_files)
            if nrrd_files:
                raw_modalities["ct"] += 1

    # -- Scan source/controls --
    controls_dir = RAW_ROOT / "source" / "controls"
    if controls_dir.exists():
        for patient_dir in sorted(controls_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name
            raw_patients.add(pid)
            for study_dir in sorted(patient_dir.iterdir()):
                if not study_dir.is_dir():
                    continue
                raw_studies += 1
                for f in study_dir.glob("*.nrrd"):
                    raw_files += 1
                    mod = _resolve_modality(
                        f.stem.split("_")[0] if "_" in f.stem else f.stem
                    )
                    raw_modalities[mod] += 1

    # -- Scan extension_1 --
    ext_dir = RAW_ROOT / "extension_1"
    if ext_dir.exists():
        for patient_dir in sorted(ext_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = f"P{patient_dir.name}"
            raw_patients.add(pid)
            for study_dir in sorted(patient_dir.iterdir()):
                if not study_dir.is_dir():
                    continue
                raw_studies += 1
                for f in study_dir.glob("*.nrrd"):
                    raw_files += 1
                    mod = _resolve_modality(
                        f.stem.split("_")[0] if "_" in f.stem else f.stem
                    )
                    raw_modalities[mod] += 1

    # Count baseline studies (each patient in RM has at least baseline)
    baseline_patients_in_rm = set()
    if rm_dir.exists():
        for modality_dir in rm_dir.iterdir():
            if modality_dir.is_dir():
                for patient_dir in modality_dir.iterdir():
                    if patient_dir.is_dir():
                        baseline_patients_in_rm.add(patient_dir.name)
    raw_studies += len(baseline_patients_in_rm)

    # ── Reorganized state (Phase 1 output) ──
    # Use the approach of scanning the ACTUAL Phase 1 output directory pattern.
    # After Phase 1, data is at curated/dataset/MenGrowth-2025/P{N}/{study_num}/
    # Some of those are now renamed (Phase 4) and some are deleted (Phases 2-3).
    # Reconstruct from: current dataset + rejected_files.csv

    # Load id_mapping (old_id -> new_id) and invert
    with open(ID_MAPPING_PATH) as f:
        id_mapping = json.load(f)

    # Track reorganized patients/studies
    reorg_patients: dict[str, set[str]] = defaultdict(set)  # pid -> set of study_ids
    reorg_modalities: Counter = Counter()  # modality -> count of studies containing it
    reorg_files = 0

    # Current curated dataset files (map back to original IDs)
    for old_pid, mapping in id_mapping.items():
        new_pid = mapping["new_id"]
        new_pid_dir = DATASET_DIR / new_pid
        if not new_pid_dir.exists():
            continue
        for study_dir in sorted(new_pid_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            # Find original study number from mapping
            new_study_id = study_dir.name
            for orig_study, mapped_study in mapping["studies"].items():
                if mapped_study == new_study_id:
                    reorg_patients[old_pid].add(orig_study)
                    break
            for f in study_dir.glob("*.nrrd"):
                reorg_files += 1
                mod = f.stem.split("-")[0] if "-" in f.stem else f.stem
                reorg_modalities[mod] += 1

    # Add back rejected files (quality/rejected_files.csv)
    rejected_quality = _load_rejected_csv(QUALITY_DIR / "rejected_files.csv")
    for row in rejected_quality:
        pid = row["patient_id"]
        study = row["study_name"]
        filename = row["filename"]
        reorg_patients[pid].add(study)
        reorg_files += 1
        mod = (
            filename.replace(".nrrd", "").split("-")[0]
            if "-" in filename
            else filename.replace(".nrrd", "")
        )
        reorg_modalities[mod] += 1

    total_reorg_patients = len(reorg_patients)
    total_reorg_studies = sum(len(studies) for studies in reorg_patients.values())

    # Studies per patient distribution
    studies_counts = [len(studies) for studies in reorg_patients.values()]
    studies_dist = Counter(studies_counts)

    # Modality availability in reorganized data (per-study counts)
    # Re-count by iterating studies
    mod_per_study: dict[str, set[str]] = defaultdict(
        set
    )  # "pid/study" -> set of modalities
    for old_pid, mapping in id_mapping.items():
        new_pid = mapping["new_id"]
        new_pid_dir = DATASET_DIR / new_pid
        if not new_pid_dir.exists():
            continue
        for study_dir in sorted(new_pid_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            new_study_id = study_dir.name
            for orig_study, mapped_study in mapping["studies"].items():
                if mapped_study == new_study_id:
                    key = f"{old_pid}/{orig_study}"
                    for f in study_dir.glob("*.nrrd"):
                        mod_per_study[key].add(f.stem)
                    break

    for row in rejected_quality:
        key = f"{row['patient_id']}/{row['study_name']}"
        mod = row["filename"].replace(".nrrd", "")
        mod_per_study[key].add(mod)

    modality_availability = {}
    for mod in ["t1c", "t1n", "t2w", "t2f", "swi", "dwi", "adc", "ct"]:
        count = sum(1 for mods in mod_per_study.values() if mod in mods)
        modality_availability[mod] = count

    all_modalities_observed = set()
    for mods in mod_per_study.values():
        all_modalities_observed.update(mods)

    result = {
        "total_patients": total_reorg_patients,
        "total_studies": total_reorg_studies,
        "total_nrrd_files": reorg_files,
        "studies_per_patient": {
            "min": min(studies_counts) if studies_counts else 0,
            "max": max(studies_counts) if studies_counts else 0,
            "mean": round(statistics.mean(studies_counts), 2) if studies_counts else 0,
            "median": round(statistics.median(studies_counts), 1)
            if studies_counts
            else 0,
            "distribution": {str(k): v for k, v in sorted(studies_dist.items())},
        },
        "modalities_observed": sorted(all_modalities_observed),
        "modality_availability_reorganized": modality_availability,
    }

    _write_json(result, OUTPUT_DIR / "raw_cohort_summary.json")
    return result


# ── 2. Clinical Metadata Summary ──────────────────────────────────────────────


def generate_clinical_metadata_summary() -> dict:
    """Generate demographics and clinical characteristics."""
    logger.info("Generating clinical_metadata_summary.json ...")

    mgr = MetadataManager()
    mgr.load_from_xlsx(XLSX_PATH)

    # Load enriched CSV for inclusion status
    df_meta = pd.read_csv(METADATA_CSV_PATH)

    # Get all patients from the manager
    all_patients = mgr.get_all_patients()

    # Use ALL patients with metadata (not just included) for the full cohort view
    # But also compute stats for included-only
    included_pids = set(df_meta.loc[df_meta["included"] == True, "patient_id"].values)

    # Age statistics (all patients with metadata)
    all_ages = [p.age for p in all_patients.values() if p.age is not None]
    included_ages = [
        p.age
        for pid, p in all_patients.items()
        if p.age is not None and pid in included_pids
    ]

    def _age_stats(ages: list) -> dict:
        if not ages:
            return {}
        bins = list(range(25, 95, 5))
        counts, _ = np.histogram(ages, bins=bins)
        return {
            "min": int(min(ages)),
            "max": int(max(ages)),
            "mean": round(statistics.mean(ages), 1),
            "median": round(statistics.median(ages), 1),
            "std": round(statistics.stdev(ages), 1) if len(ages) > 1 else 0,
            "histogram_bins": bins,
            "histogram_counts": counts.tolist(),
        }

    # Sex distribution
    def _sex_stats(patients: dict, pid_filter: set | None = None) -> dict:
        counts = {"male": 0, "female": 0, "unknown": 0}
        for pid, p in patients.items():
            if pid_filter and pid not in pid_filter:
                continue
            if p.sex == 0:
                counts["male"] += 1
            elif p.sex == 1:
                counts["female"] += 1
            else:
                counts["unknown"] += 1
        return counts

    # Growth status
    def _growth_stats(patients: dict, pid_filter: set | None = None) -> dict:
        counts = {"growing": 0, "stable": 0, "unknown": 0}
        for pid, p in patients.items():
            if pid_filter and pid not in pid_filter:
                continue
            g = p.get_growth_status()
            if g is True:
                counts["growing"] += 1
            elif g is False:
                counts["stable"] += 1
            else:
                counts["unknown"] += 1
        return counts

    # Volume statistics
    def _volume_stats(patients: dict, pid_filter: set | None = None) -> dict:
        vols = []
        for pid, p in patients.items():
            if pid_filter and pid not in pid_filter:
                continue
            v = p.get_first_study_volume()
            if v is not None:
                vols.append(v)
        if not vols:
            return {}
        vols_arr = np.array(vols)
        return {
            "n": len(vols),
            "min": round(float(np.min(vols_arr)), 1),
            "max": round(float(np.max(vols_arr)), 1),
            "mean": round(float(np.mean(vols_arr)), 1),
            "median": round(float(np.median(vols_arr)), 1),
            "std": round(float(np.std(vols_arr, ddof=1)), 1) if len(vols) > 1 else 0,
            "q1": round(float(np.percentile(vols_arr, 25)), 1),
            "q3": round(float(np.percentile(vols_arr, 75)), 1),
        }

    # Scanner distribution
    def _scanner_stats(patients: dict, pid_filter: set | None = None) -> dict:
        counts: Counter = Counter()
        for pid, p in patients.items():
            if pid_filter and pid not in pid_filter:
                continue
            scanner = p.scanner or "unknown"
            counts[scanner] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # Controls per patient (from metadata, reflects clinical follow-ups not MRI studies)
    def _controls_stats(patients: dict, pid_filter: set | None = None) -> dict:
        nums = []
        for pid, p in patients.items():
            if pid_filter and pid not in pid_filter:
                continue
            nums.append(p.get_num_controls())
        if not nums:
            return {}
        dist = Counter(nums)
        return {
            "min": min(nums),
            "max": max(nums),
            "mean": round(statistics.mean(nums), 1),
            "distribution": {str(k): v for k, v in sorted(dist.items())},
        }

    result = {
        "all_patients_with_metadata": {
            "count": len(all_patients),
            "age": _age_stats(all_ages),
            "sex": _sex_stats(all_patients),
            "growth_status": _growth_stats(all_patients),
            "initial_volume_mm3": _volume_stats(all_patients),
            "scanner_distribution": _scanner_stats(all_patients),
            "num_controls_per_patient": _controls_stats(all_patients),
        },
        "included_patients": {
            "count": len(included_pids),
            "age": _age_stats(included_ages),
            "sex": _sex_stats(all_patients, included_pids),
            "growth_status": _growth_stats(all_patients, included_pids),
            "initial_volume_mm3": _volume_stats(all_patients, included_pids),
            "scanner_distribution": _scanner_stats(all_patients, included_pids),
            "num_controls_per_patient": _controls_stats(all_patients, included_pids),
        },
    }

    _write_json(result, OUTPUT_DIR / "clinical_metadata_summary.json")
    return result


# ── 3. Pipeline Attrition ─────────────────────────────────────────────────────


def generate_pipeline_attrition() -> dict:
    """Generate patient/study counts at each pipeline stage."""
    logger.info("Generating pipeline_attrition.json ...")

    # Load rejected files from both CSVs
    rejects_stage0 = _load_rejected_csv(CURATED_ROOT / "dataset" / "rejected_files.csv")
    rejects_quality = _load_rejected_csv(QUALITY_DIR / "rejected_files.csv")

    # Load id_mapping
    with open(ID_MAPPING_PATH) as f:
        id_mapping = json.load(f)

    # ── Stage 0: Raw data ──
    # Count distinct patients mentioned in metadata
    df_meta = pd.read_csv(METADATA_CSV_PATH)
    all_raw_patients = set(df_meta["patient_id"].values)
    # Also add patients from extension that might not have metadata rows
    for patient_dir in (RAW_ROOT / "extension_1").iterdir():
        if patient_dir.is_dir():
            all_raw_patients.add(f"P{patient_dir.name}")

    # ── Stage 1: After reorganization ──
    # Reconstruct reorganized state: current dataset (mapped back) + all quality rejected
    reorg_data = _reconstruct_reorganized_state(id_mapping, rejects_quality)
    reorg_patients = set(reorg_data.keys())
    reorg_studies = sum(len(studies) for studies in reorg_data.values())
    reorg_files = sum(
        sum(len(mods) for mods in studies.values()) for studies in reorg_data.values()
    )

    # Stage 0 rejection reasons (reorganization exclusions)
    stage0_reasons: Counter = Counter()
    for row in rejects_stage0:
        reason = row["rejection_reason"]
        if "exclusion pattern" in reason.lower():
            stage0_reasons["exclusion_pattern"] += 1
        else:
            stage0_reasons[reason] += 1

    # ── Stage 2: After completeness filtering ──
    # Parse quality/rejected_files.csv for completeness reasons (filtering_study, filtering_patient)
    completeness_rejected_studies: set = set()  # "pid/study" keys
    completeness_rejected_patients: set = set()
    quality_rejected_studies: set = set()
    quality_rejected_patients: set = set()

    # Categorize rejections
    completeness_reasons: Counter = Counter()
    quality_reasons: Counter = Counter()

    for row in rejects_quality:
        pid = row["patient_id"]
        study = row["study_name"]
        reason = row["rejection_reason"]
        source_type = row["source_type"]
        stage = row["stage"]

        key = f"{pid}/{study}"

        if "missing" in reason.lower() and "sequences" in reason.lower():
            completeness_rejected_studies.add(key)
            completeness_reasons["missing_modalities"] += 1
        elif "valid studies" in reason.lower() or "minimum required" in reason.lower():
            # Patient removed due to too few valid studies
            if source_type == "filtering_patient":
                completeness_rejected_patients.add(pid)
                completeness_reasons["insufficient_timepoints"] += 1
            elif source_type == "quality_filtering":
                quality_rejected_patients.add(pid)
            else:
                completeness_rejected_patients.add(pid)
                completeness_reasons["insufficient_timepoints"] += 1
        elif "quality_filter" in str(reason) or source_type == "quality_filtering":
            quality_rejected_studies.add(key)
            # Parse specific quality reasons
            if "ghosting" in reason.lower():
                quality_reasons["B5_ghosting_detection"] += 1
            if "motion" in reason.lower():
                quality_reasons["B4_motion_artifact"] += 1
            if "brain_coverage" in reason.lower():
                quality_reasons["C4_brain_coverage"] += 1
            if "fov" in reason.lower():
                quality_reasons["C2_fov_consistency"] += 1
            if "affine" in reason.lower():
                quality_reasons["C1_affine_validation"] += 1
            if "nrrd" in reason.lower():
                quality_reasons["A1_nrrd_header"] += 1
            if "scout" in reason.lower():
                quality_reasons["A2_scout_detection"] += 1
            if "snr" in reason.lower():
                quality_reasons["B1_snr"] += 1
            if "contrast" in reason.lower():
                quality_reasons["B2_contrast"] += 1
            if "registration" in reason.lower():
                quality_reasons["E1_registration_ref"] += 1

    # Also parse quality_issues.csv for detailed per-check blocking reasons
    quality_issues = _load_quality_issues()

    # Count unique studies blocked by quality issues
    blocked_studies_by_check: Counter = Counter()
    for issue in quality_issues:
        if issue["action"] == "block":
            check = issue["check_name"]
            blocked_studies_by_check[check] += 1

    # Compute stage counts
    # Stage 2 = reorganized - completeness_rejected
    completeness_unique_studies = set()
    for row in rejects_quality:
        reason = row["rejection_reason"]
        if "missing" in reason.lower() and "sequences" in reason.lower():
            completeness_unique_studies.add(f"{row['patient_id']}/{row['study_name']}")

    # Patients removed at completeness stage
    # A patient is "removed at completeness" if ALL their studies were removed
    # and they appear as filtering_patient
    patients_after_completeness = set()
    studies_after_completeness = 0
    files_after_completeness = 0
    for pid, studies in reorg_data.items():
        pid_has_valid_study = False
        for study_id, mods in studies.items():
            key = f"{pid}/{study_id}"
            if key not in completeness_unique_studies:
                pid_has_valid_study = True
                studies_after_completeness += 1
                files_after_completeness += len(mods)
        if pid_has_valid_study and pid not in completeness_rejected_patients:
            patients_after_completeness.add(pid)

    # Stage 3 = after quality filtering
    # Parse quality_issues.csv for blocking checks, then remove affected studies/patients
    # First, find which studies have ANY blocking issue
    blocked_study_keys: set = set()
    for issue in quality_issues:
        if issue["action"] == "block":
            blocked_study_keys.add(f"{issue['patient_id']}/{issue['study_id']}")

    studies_after_quality = 0
    patients_after_quality_data: dict[str, int] = defaultdict(int)
    for pid in patients_after_completeness:
        for study_id, mods in reorg_data.get(pid, {}).items():
            key = f"{pid}/{study_id}"
            if key in completeness_unique_studies:
                continue
            if key not in blocked_study_keys:
                studies_after_quality += 1
                patients_after_quality_data[pid] += 1

    # Apply min_studies filter (patients need >= 2 valid studies)
    patients_after_quality = {
        pid for pid, count in patients_after_quality_data.items() if count >= 2
    }

    # Final curated state
    final_patients = len(id_mapping)
    final_studies = sum(len(m["studies"]) for m in id_mapping.values())
    final_files = 0
    modalities_per_study_counts: Counter = Counter()
    for old_pid, mapping in id_mapping.items():
        for orig_study, new_study in mapping["studies"].items():
            study_dir = DATASET_DIR / mapping["new_id"] / new_study
            if study_dir.exists():
                nrrd_count = len(list(study_dir.glob("*.nrrd")))
                final_files += nrrd_count
                modalities_per_study_counts[nrrd_count] += 1

    result = {
        "stage_0_raw": {
            "description": "Initial raw data before any processing",
            "patients": len(all_raw_patients),
        },
        "stage_1_reorganized": {
            "description": "After Phase 1 reorganization (standardized directory layout)",
            "patients": len(reorg_patients),
            "studies": reorg_studies,
            "files": reorg_files,
            "files_excluded_during_reorg": len(rejects_stage0),
            "exclusion_reasons": dict(stage0_reasons),
        },
        "stage_2_completeness_filtered": {
            "description": "After Phase 2 completeness filtering (required modalities + min timepoints)",
            "patients": len(patients_after_completeness),
            "studies": studies_after_completeness,
            "files": files_after_completeness,
            "patients_lost": len(reorg_patients) - len(patients_after_completeness),
            "studies_lost": reorg_studies - studies_after_completeness,
            "loss_reasons": dict(completeness_reasons),
        },
        "stage_3_quality_filtered": {
            "description": "After Phase 3 quality filtering (15 automated checks)",
            "patients": len(patients_after_quality),
            "studies": studies_after_quality,
            "patients_lost_at_quality": len(patients_after_completeness)
            - len(patients_after_quality),
            "studies_blocked_by_check": dict(blocked_studies_by_check),
            "quality_issues_detail": dict(quality_reasons),
        },
        "stage_4_final_curated": {
            "description": "Final curated dataset after re-identification",
            "patients": final_patients,
            "studies": final_studies,
            "files": final_files,
            "modalities_per_study": {
                f"{k}_modalities": v
                for k, v in sorted(modalities_per_study_counts.items())
            },
        },
    }

    _write_json(result, OUTPUT_DIR / "pipeline_attrition.json")
    return result


# ── 4. Quality Check Results CSV ──────────────────────────────────────────────


def generate_quality_check_results() -> None:
    """Parse quality_metrics.json for per-file quality check results."""
    logger.info("Generating quality_check_results.csv ...")

    metrics_path = QUALITY_DIR / "quality_metrics.json"
    if not metrics_path.exists():
        logger.warning(f"quality_metrics.json not found at {metrics_path}")
        return

    # Stream-parse the large JSON
    with open(metrics_path) as f:
        metrics = json.load(f)

    rows = []
    for pid, patient_data in sorted(metrics.get("patients", {}).items()):
        for study_id, study_data in sorted(patient_data.get("studies", {}).items()):
            for modality, mod_data in sorted(study_data.get("files", {}).items()):
                checks = mod_data.get("checks", {})
                for check_name, check_result in sorted(checks.items()):
                    passed = check_result.get("passed", True)
                    action = check_result.get("action", "warn")
                    message = check_result.get("message", "")
                    details = check_result.get("details", {})

                    # Extract key metric value based on check type
                    metric_value = ""
                    threshold = ""
                    if check_name == "motion_artifact":
                        metric_value = details.get("gradient_entropy", "")
                        threshold = details.get("threshold", "")
                    elif check_name == "ghosting_detection":
                        metric_value = details.get("corner_foreground_ratio", "")
                        threshold = "0.15"
                    elif check_name == "snr_filtering":
                        metric_value = details.get("snr", "")
                        threshold = details.get("threshold", "")
                    elif check_name == "intensity_outliers":
                        if "max" in details and "p99" in details:
                            p99 = details["p99"]
                            mx = details["max"]
                            metric_value = round(mx / p99, 2) if p99 > 0 else ""
                        threshold = details.get("threshold", "")
                    elif check_name == "brain_coverage":
                        metric_value = details.get("min_extent", "")
                        threshold = "100.0"
                    elif check_name == "fov_consistency":
                        metric_value = details.get("ratio", "")
                        threshold = "5.0 (block) / 3.0 (warn)"
                    elif check_name == "voxel_spacing":
                        metric_value = details.get("anisotropy", "")
                        threshold = "20.0"
                    elif check_name == "contrast_detection":
                        metric_value = details.get("std_ratio", "")
                        threshold = "0.10"

                    rows.append(
                        {
                            "patient_id": pid,
                            "study_id": study_id,
                            "modality": modality,
                            "check_name": check_name,
                            "passed": passed,
                            "action": action,
                            "metric_value": metric_value,
                            "threshold": threshold,
                            "message": message,
                        }
                    )

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "quality_check_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"  Written {len(rows)} rows to {output_path}")


# ── 5. Spacing Summary CSV ────────────────────────────────────────────────────


def generate_spacing_summary() -> None:
    """Extract voxel spacing from per_study_metrics.csv."""
    logger.info("Generating spacing_summary.csv ...")

    metrics_path = QC_DIR / "per_study_metrics.csv"
    if not metrics_path.exists():
        logger.warning(f"per_study_metrics.csv not found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)

    # Select relevant columns
    cols = [
        "patient_id",
        "study_id",
        "sequence",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "width",
        "height",
        "depth",
    ]
    available_cols = [c for c in cols if c in df.columns]
    df_out = df[available_cols].copy()
    df_out = df_out.rename(
        columns={
            "sequence": "modality",
            "width": "dim_x",
            "height": "dim_y",
            "depth": "dim_z",
        }
    )

    output_path = OUTPUT_DIR / "spacing_summary.csv"
    df_out.to_csv(output_path, index=False)
    logger.info(f"  Written {len(df_out)} rows to {output_path}")


# ── 6. Rejected Files Summary CSV ────────────────────────────────────────────


def generate_rejected_files_summary() -> None:
    """Aggregate rejection counts by stage and reason."""
    logger.info("Generating rejected_files_summary.csv ...")

    rows = []

    # Stage 0: Reorganization exclusions
    rejects_stage0 = _load_rejected_csv(CURATED_ROOT / "dataset" / "rejected_files.csv")
    stage0_reasons: Counter = Counter()
    for row in rejects_stage0:
        reason = row["rejection_reason"]
        # Normalize: extract the pattern from "Matched exclusion pattern: *seg*.nrrd"
        if "exclusion pattern" in reason.lower():
            pattern = reason.split(":")[-1].strip() if ":" in reason else reason
            stage0_reasons[f"exclusion_pattern:{pattern}"] += 1
        else:
            stage0_reasons[reason] += 1

    for reason, count in sorted(stage0_reasons.items()):
        rows.append(
            {
                "stage": 0,
                "stage_name": "reorganization",
                "reason": reason,
                "count": count,
            }
        )

    # Stages 1-2: Completeness + Quality filtering
    rejects_quality = _load_rejected_csv(QUALITY_DIR / "rejected_files.csv")

    # Categorize
    stage1_reasons: Counter = Counter()
    stage2_reasons: Counter = Counter()

    for row in rejects_quality:
        reason = row["rejection_reason"]
        source_type = row["source_type"]

        if "missing" in reason.lower() and "sequences" in reason.lower():
            stage1_reasons["missing_modalities"] += 1
        elif "valid studies" in reason.lower() or "minimum required" in reason.lower():
            if source_type == "quality_filtering":
                stage2_reasons["insufficient_timepoints_after_quality"] += 1
            else:
                stage1_reasons["insufficient_timepoints"] += 1
        elif "quality_filter" in str(reason):
            # Parse individual quality reasons
            checks = reason.replace("quality_filter:", "").split(",")
            for check in checks:
                stage2_reasons[check.strip()] += 1
        else:
            stage1_reasons[reason] += 1

    for reason, count in sorted(stage1_reasons.items()):
        rows.append(
            {
                "stage": 1,
                "stage_name": "completeness_filtering",
                "reason": reason,
                "count": count,
            }
        )

    for reason, count in sorted(stage2_reasons.items()):
        rows.append(
            {
                "stage": 2,
                "stage_name": "quality_filtering",
                "reason": reason,
                "count": count,
            }
        )

    # Also add per-check blocking counts from quality_issues.csv
    quality_issues = _load_quality_issues()
    check_block_counts: Counter = Counter()
    for issue in quality_issues:
        if issue["action"] == "block":
            check_block_counts[issue["check_name"]] += 1

    for check, count in sorted(check_block_counts.items()):
        rows.append(
            {
                "stage": 2,
                "stage_name": "quality_filtering_detail",
                "reason": f"blocking_issue:{check}",
                "count": count,
            }
        )

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "rejected_files_summary.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"  Written {len(rows)} rows to {output_path}")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_rejected_csv(path: Path) -> list[dict]:
    """Load a rejected_files.csv with proper quoting."""
    rows = []
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_quality_issues() -> list[dict]:
    """Load quality_issues.csv."""
    path = QUALITY_DIR / "quality_issues.csv"
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _reconstruct_reorganized_state(
    id_mapping: dict, rejects_quality: list[dict]
) -> dict[str, dict[str, list[str]]]:
    """Reconstruct Phase 1 output: pid -> {study_id -> [modalities]}.

    Combines current curated dataset (mapped back to original IDs) with
    rejected files from quality/rejected_files.csv.
    """
    data: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    # Current dataset (map back to original IDs)
    for old_pid, mapping in id_mapping.items():
        new_pid = mapping["new_id"]
        new_pid_dir = DATASET_DIR / new_pid
        if not new_pid_dir.exists():
            continue
        for orig_study, new_study in mapping["studies"].items():
            study_dir = new_pid_dir / new_study
            if study_dir.exists():
                for f in sorted(study_dir.glob("*.nrrd")):
                    data[old_pid][orig_study].append(f.stem)

    # Rejected files
    for row in rejects_quality:
        pid = row["patient_id"]
        study = row["study_name"]
        filename = row["filename"]
        mod = filename.replace(".nrrd", "")
        if mod not in data[pid][study]:
            data[pid][study].append(mod)

    return dict(data)


def _write_json(data: dict, path: Path) -> None:
    """Write dict as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Written {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    generate_raw_cohort_summary()
    generate_clinical_metadata_summary()
    generate_pipeline_attrition()
    generate_quality_check_results()
    generate_spacing_summary()
    generate_rejected_files_summary()

    logger.info("=" * 60)
    logger.info("All thesis data files generated successfully!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        logger.info(f"  {f.name:40s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
