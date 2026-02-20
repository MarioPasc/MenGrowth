"""Manual curation of the curated dataset.

Provides functionality to remove specific studies from the curated dataset
based on a YAML configuration file, with cascade checks for minimum
longitudinal studies per patient.

Usage:
    mengrowth-curate-dataset --final-manual-curation configs/templates/manual_curation.yaml
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from mengrowth.preprocessing.utils.filter_raw_data import append_to_rejected_files_csv
from mengrowth.preprocessing.utils.reorganize_raw_data import RejectedFile

logger = logging.getLogger(__name__)

MANUAL_CURATION_STAGE = 3


@dataclass
class ManualExclusion:
    """A single study exclusion entry.

    Attributes:
        patient_id: MenGrowth patient ID (e.g., "MenGrowth-0034").
        study_id: MenGrowth study ID (e.g., "MenGrowth-0034-000").
        reason: Human-readable reason for exclusion.
        sequence: Modalities where the issue was observed (e.g., ["t1c", "t2w"]).
            Informational only — the entire study is always removed.
    """

    patient_id: str
    study_id: str
    reason: str
    sequence: List[str] = field(default_factory=list)


@dataclass
class ManualCurationConfig:
    """Configuration for manual curation.

    Attributes:
        exclusions: List of study exclusions to apply.
        min_studies_per_patient: Minimum longitudinal studies required to keep
            a patient after exclusions are applied.
    """

    exclusions: List[ManualExclusion] = field(default_factory=list)
    min_studies_per_patient: int = 2


@dataclass
class ManualCurationStats:
    """Statistics from a manual curation run.

    Attributes:
        studies_removed: Number of studies removed by explicit exclusion.
        patients_cascade_removed: Number of patients removed due to
            insufficient remaining studies.
        patients_remaining: Number of patients remaining after curation.
        studies_remaining: Number of studies remaining after curation.
        exclusions_skipped: Number of exclusions that referenced
            non-existent studies (already removed or typo).
    """

    studies_removed: int = 0
    patients_cascade_removed: int = 0
    patients_remaining: int = 0
    studies_remaining: int = 0
    exclusions_skipped: int = 0


def load_manual_curation_config(yaml_path: Path) -> ManualCurationConfig:
    """Load manual curation configuration from YAML.

    Args:
        yaml_path: Path to the manual curation YAML file.

    Returns:
        Parsed ManualCurationConfig.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If YAML structure is invalid.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Manual curation config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty manual curation config: {yaml_path}")

    exclusions_raw = raw.get("exclusions", [])
    if not isinstance(exclusions_raw, list):
        raise ValueError(
            f"'exclusions' must be a list, got {type(exclusions_raw).__name__}"
        )

    exclusions = []
    for i, entry in enumerate(exclusions_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Exclusion entry {i} must be a dict, got {type(entry).__name__}"
            )

        missing = [k for k in ("patient_id", "study_id", "reason") if k not in entry]
        if missing:
            raise ValueError(f"Exclusion entry {i} missing required fields: {missing}")

        # Parse optional sequence field (list of modality strings)
        raw_seq = entry.get("sequence", [])
        if isinstance(raw_seq, str):
            raw_seq = [raw_seq]
        sequence = [str(s) for s in raw_seq] if raw_seq else []

        exclusions.append(
            ManualExclusion(
                patient_id=str(entry["patient_id"]),
                study_id=str(entry["study_id"]),
                reason=str(entry["reason"]),
                sequence=sequence,
            )
        )

    min_studies = raw.get("min_studies_per_patient", 2)

    logger.info(
        f"Loaded manual curation config: {len(exclusions)} exclusions, "
        f"min_studies_per_patient={min_studies}"
    )

    return ManualCurationConfig(
        exclusions=exclusions,
        min_studies_per_patient=min_studies,
    )


def apply_manual_curation(
    config: ManualCurationConfig,
    mengrowth_dir: Path,
    quality_dir: Path,
) -> ManualCurationStats:
    """Apply manual curation exclusions to the curated dataset.

    For each exclusion, removes the study directory and logs the rejection.
    After all explicit removals, checks each affected patient for minimum
    study count and removes patients that fall below the threshold.

    Args:
        config: Manual curation configuration with exclusions.
        mengrowth_dir: Path to MenGrowth-2025 dataset directory.
        quality_dir: Path to quality output directory (for rejected_files.csv).

    Returns:
        ManualCurationStats with removal counts.
    """
    stats = ManualCurationStats()
    rejected_files: List[RejectedFile] = []
    affected_patients: Dict[str, List[str]] = {}  # patient_id -> [removed study_ids]

    # ── Phase 1: Apply explicit study exclusions ──
    logger.info(f"Applying {len(config.exclusions)} manual exclusions ...")

    for exclusion in config.exclusions:
        study_dir = mengrowth_dir / exclusion.patient_id / exclusion.study_id

        if not study_dir.exists():
            logger.warning(
                f"Study directory not found, skipping: {study_dir} "
                f"(reason: {exclusion.reason})"
            )
            stats.exclusions_skipped += 1
            continue

        # Build rejection reason with sequence info
        seq_tag = f" [{', '.join(exclusion.sequence)}]" if exclusion.sequence else ""
        rejection_reason = f"Manual curation: {exclusion.reason}{seq_tag}"

        # Collect files for rejection records
        nrrd_files = sorted(study_dir.glob("*.nrrd"))
        nifti_files = sorted(study_dir.glob("*.nii.gz"))
        all_files = nrrd_files + nifti_files

        for file_path in all_files:
            rejected_files.append(
                RejectedFile(
                    source_path=str(file_path),
                    filename=file_path.name,
                    patient_id=exclusion.patient_id,
                    study_name=exclusion.study_id,
                    rejection_reason=rejection_reason,
                    source_type="manual_curation",
                    stage=MANUAL_CURATION_STAGE,
                )
            )

        # Remove study directory
        shutil.rmtree(study_dir)
        stats.studies_removed += 1
        logger.info(
            f"  Removed {exclusion.study_id} from {exclusion.patient_id}: "
            f"{exclusion.reason}{seq_tag} ({len(all_files)} files)"
        )

        # Track affected patients
        affected_patients.setdefault(exclusion.patient_id, []).append(
            exclusion.study_id
        )

    # ── Phase 2: Cascade check — remove patients with too few studies ──
    logger.info(
        f"Checking min_studies_per_patient={config.min_studies_per_patient} "
        f"for {len(affected_patients)} affected patients ..."
    )

    for patient_id in sorted(affected_patients.keys()):
        patient_dir = mengrowth_dir / patient_id
        if not patient_dir.exists():
            continue

        remaining_studies = sorted(
            [d for d in patient_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        n_remaining = len(remaining_studies)

        if n_remaining < config.min_studies_per_patient:
            logger.info(
                f"  Patient {patient_id} has {n_remaining} studies remaining "
                f"(min={config.min_studies_per_patient}), removing patient"
            )

            # Create rejection records for remaining files
            for study_dir in remaining_studies:
                nrrd_files = sorted(study_dir.glob("*.nrrd"))
                nifti_files = sorted(study_dir.glob("*.nii.gz"))
                all_files = nrrd_files + nifti_files

                for file_path in all_files:
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name=study_dir.name,
                            rejection_reason=(
                                f"Insufficient studies after manual curation "
                                f"({n_remaining} remaining, min={config.min_studies_per_patient})"
                            ),
                            source_type="manual_curation",
                            stage=MANUAL_CURATION_STAGE,
                        )
                    )

            shutil.rmtree(patient_dir)
            stats.patients_cascade_removed += 1
        else:
            logger.info(f"  Patient {patient_id}: {n_remaining} studies remaining (OK)")

    # ── Phase 3: Log rejections ──
    if rejected_files:
        rejected_csv = quality_dir / "rejected_files.csv"
        append_to_rejected_files_csv(rejected_files, rejected_csv)
        logger.info(f"Logged {len(rejected_files)} rejection records to {rejected_csv}")

    # ── Phase 4: Compute final stats ──
    remaining_patients = sorted(
        [d for d in mengrowth_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    stats.patients_remaining = len(remaining_patients)
    stats.studies_remaining = sum(
        len([s for s in p.iterdir() if s.is_dir()]) for p in remaining_patients
    )

    logger.info(
        f"Manual curation complete: "
        f"{stats.studies_removed} studies removed, "
        f"{stats.patients_cascade_removed} patients cascade-removed, "
        f"{stats.exclusions_skipped} exclusions skipped. "
        f"Dataset: {stats.patients_remaining} patients, "
        f"{stats.studies_remaining} studies"
    )

    return stats


def reid_after_manual_curation(
    mengrowth_dir: Path,
    id_mapping_path: Path,
) -> Dict[str, str]:
    """Re-number MenGrowth IDs to close gaps after patient removal.

    Reads the existing id_mapping.json, renumbers the remaining patient
    directories so that MenGrowth IDs are continuous (no gaps), updates
    study directory names accordingly, and writes the updated mapping.

    Args:
        mengrowth_dir: Path to MenGrowth-2025 dataset directory.
        id_mapping_path: Path to id_mapping.json (read + overwritten).

    Returns:
        Dictionary mapping old MenGrowth IDs to new MenGrowth IDs
        (only entries that changed). Empty dict if no renumbering needed.
    """
    import json

    # ── Load existing id_mapping ──
    if not id_mapping_path.exists():
        logger.warning(
            f"id_mapping.json not found at {id_mapping_path}, skipping re-ID"
        )
        return {}

    with open(id_mapping_path, "r", encoding="utf-8") as f:
        old_mapping = json.load(f)

    # ── Get remaining patient directories (already MenGrowth-XXXX format) ──
    patient_dirs = sorted(
        [d for d in mengrowth_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    # Build reverse lookup: MenGrowth-XXXX → original P{N}
    mengrowth_to_original: Dict[str, str] = {}
    for original_id, info in old_mapping.items():
        mengrowth_to_original[info["new_id"]] = original_id

    # ── Check if renumbering is needed ──
    expected_ids = [f"MenGrowth-{i + 1:04d}" for i in range(len(patient_dirs))]
    actual_ids = [d.name for d in patient_dirs]

    if expected_ids == actual_ids:
        logger.info("MenGrowth IDs are already continuous, no re-ID needed")
        return {}

    # ── Rename to temporary names first (avoid collisions) ──
    temp_names: Dict[str, str] = {}
    for patient_dir in patient_dirs:
        temp_name = f"_temp_{patient_dir.name}"
        temp_path = mengrowth_dir / temp_name
        patient_dir.rename(temp_path)
        temp_names[patient_dir.name] = temp_name

    # ── Rename to final continuous IDs ──
    rename_map: Dict[str, str] = {}  # old MenGrowth ID → new MenGrowth ID
    new_mapping: Dict[str, Dict] = {}

    for new_idx, old_id in enumerate(sorted(temp_names.keys()), start=1):
        new_patient_id = f"MenGrowth-{new_idx:04d}"
        temp_name = temp_names[old_id]
        temp_path = mengrowth_dir / temp_name
        new_patient_path = mengrowth_dir / new_patient_id

        # Rename patient directory
        temp_path.rename(new_patient_path)

        if old_id != new_patient_id:
            rename_map[old_id] = new_patient_id
            logger.info(f"  Re-ID: {old_id} → {new_patient_id}")

        # Rename study directories within patient
        study_dirs = sorted(
            [d for d in new_patient_path.iterdir() if d.is_dir()],
            key=lambda s: s.name,
        )

        study_mappings = {}
        for study_idx, study_dir in enumerate(study_dirs):
            old_study_name = study_dir.name
            new_study_id = f"{new_patient_id}-{study_idx:03d}"

            if old_study_name != new_study_id:
                new_study_path = new_patient_path / new_study_id
                study_dir.rename(new_study_path)

            # Find original study key from old mapping
            original_patient_id = mengrowth_to_original.get(old_id, old_id)
            old_patient_info = old_mapping.get(original_patient_id, {})
            old_studies = old_patient_info.get("studies", {})

            # Reverse lookup: find original study number from old MenGrowth study ID
            original_study_key = None
            for orig_key, mapped_study_id in old_studies.items():
                if mapped_study_id == old_study_name:
                    original_study_key = orig_key
                    break

            if original_study_key is not None:
                study_mappings[original_study_key] = new_study_id
            else:
                study_mappings[str(study_idx)] = new_study_id

        # Update mapping
        original_patient_id = mengrowth_to_original.get(old_id, old_id)
        new_mapping[original_patient_id] = {
            "new_id": new_patient_id,
            "studies": study_mappings,
        }

    # ── Write updated id_mapping.json ──
    with open(id_mapping_path, "w", encoding="utf-8") as f:
        json.dump(new_mapping, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Re-ID complete: {len(rename_map)} patients renumbered, "
        f"id_mapping.json updated at {id_mapping_path}"
    )

    return rename_map
