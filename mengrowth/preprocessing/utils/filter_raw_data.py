"""Filter reorganized MRI data based on study completeness and quality criteria.

This module provides utilities to filter reorganized MRI data to ensure studies
meet minimum completeness requirements, patients have sufficient longitudinal data,
and sequences are standardized by applying orientation priority rules.

The filtering operates on reorganized data structure:
    {data_root}/MenGrowth-2025/P{patient_id}/{study_number}/*.nrrd

Filtering criteria:
    - Required sequences (with allowed missing threshold)
    - Minimum studies per patient
    - Orientation priority for sequence selection
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from mengrowth.preprocessing.config import FilteringConfig, PreprocessingConfig
from mengrowth.preprocessing.utils.reorganize_raw_data import RejectedFile

if TYPE_CHECKING:
    from mengrowth.preprocessing.utils.metadata import MetadataManager

# Configure module logger
logger = logging.getLogger(__name__)


def scan_study_sequences(study_path: Path) -> Dict[str, Path]:
    """Scan a study directory and return available sequences.

    Args:
        study_path: Path to study directory (e.g., .../P1/0/).

    Returns:
        Dictionary mapping sequence names (without .nrrd extension) to file paths.
        E.g., {'t1c-axial': Path(...), 't2f': Path(...), 't1n': Path(...)}

    Examples:
        >>> scan_study_sequences(Path('/data/MenGrowth-2025/P1/0'))
        {'t1c-axial': Path('/data/MenGrowth-2025/P1/0/t1c-axial.nrrd'), 't2f': ...}
    """
    sequences = {}

    if not study_path.exists() or not study_path.is_dir():
        logger.warning(f"Study path does not exist or is not a directory: {study_path}")
        return sequences

    for nrrd_file in study_path.glob("*.nrrd"):
        sequence_name = nrrd_file.stem  # Remove .nrrd extension
        sequences[sequence_name] = nrrd_file

    return sequences


def apply_orientation_priority(
    sequences: Dict[str, Path], required_seq: str, priority: List[str]
) -> Optional[Path]:
    """Find the best matching sequence file using orientation priority.

    Given a required sequence name and available sequences, find the best match
    according to the orientation priority list.

    Args:
        sequences: Available sequences in the study (name -> path mapping).
        required_seq: Required sequence name (e.g., 't1c', 't2f').
        priority: Orientation priority list (e.g., ['none', 'axial', 'sagital', 'coronal']).
            'none' means exact match without orientation suffix.

    Returns:
        Path to the best matching sequence file, or None if no match found.

    Examples:
        >>> seqs = {'t1c-axial': Path('t1c-axial.nrrd'), 't1c-sagital': Path('t1c-sagital.nrrd')}
        >>> apply_orientation_priority(seqs, 't1c', ['none', 'axial', 'sagital'])
        Path('t1c-axial.nrrd')

        >>> seqs = {'t1c': Path('t1c.nrrd'), 't1c-axial': Path('t1c-axial.nrrd')}
        >>> apply_orientation_priority(seqs, 't1c', ['none', 'axial'])
        Path('t1c.nrrd')
    """
    # Try each priority level in order
    for orientation in priority:
        if orientation == "none":
            # Exact match (no orientation suffix)
            if required_seq in sequences:
                return sequences[required_seq]
        else:
            # Match with orientation suffix
            seq_with_orientation = f"{required_seq}-{orientation}"
            if seq_with_orientation in sequences:
                return sequences[seq_with_orientation]

    return None


def normalize_study_sequences(
    study_path: Path,
    required_sequences: List[str],
    orientation_priority: List[str],
    dry_run: bool,
) -> Tuple[Set[str], List[RejectedFile]]:
    """Normalize sequences in a study by applying orientation priority.

    This function ensures each required sequence exists by:
    1. Checking if the exact sequence name exists
    2. If not, finding the best match using orientation priority
    3. Renaming the matched file to the required sequence name

    Args:
        study_path: Path to study directory.
        required_sequences: List of required sequence names.
        orientation_priority: Priority order for selecting orientations.
        dry_run: If True, simulate without making changes.

    Returns:
        Tuple of:
            - Set of normalized sequence names available in the study
            - List of RejectedFile records for deleted oriented files

    Examples:
        If study has t1c-axial.nrrd but not t1c.nrrd, and priority is ['none', 'axial'],
        it will rename t1c-axial.nrrd -> t1c.nrrd
    """
    patient_id = study_path.parent.name  # e.g., 'P1'
    study_num = study_path.name  # e.g., '0'

    sequences = scan_study_sequences(study_path)
    normalized_sequences = set()
    rejected_files = []

    for required_seq in required_sequences:
        # Check if exact match already exists
        if required_seq in sequences:
            normalized_sequences.add(required_seq)
            continue

        # Find best match using orientation priority
        best_match_path = apply_orientation_priority(
            sequences, required_seq, orientation_priority
        )

        if best_match_path:
            # Rename the file to remove orientation suffix
            target_path = study_path / f"{required_seq}.nrrd"
            original_name = best_match_path.name

            logger.debug(
                f"Normalizing {patient_id}/{study_num}: "
                f"{original_name} -> {required_seq}.nrrd"
            )

            if not dry_run:
                best_match_path.rename(target_path)

            normalized_sequences.add(required_seq)

            # Update sequences dict to reflect the rename
            sequences[required_seq] = target_path
            # Remove old entry
            old_key = best_match_path.stem
            if old_key in sequences:
                del sequences[old_key]

    # Return the set of sequences that are now available
    # (both originally present and newly normalized)
    return normalized_sequences, rejected_files


def filter_study(
    patient_id: str,
    study_num: str,
    study_path: Path,
    config: FilteringConfig,
    dry_run: bool,
) -> Tuple[bool, List[RejectedFile]]:
    """Check if a study meets filtering criteria.

    Args:
        patient_id: Patient ID (e.g., 'P1').
        study_num: Study number (e.g., '0', '1').
        study_path: Path to study directory.
        config: Filtering configuration.
        dry_run: If True, simulate without making changes.

    Returns:
        Tuple of:
            - Boolean indicating if study passes filter
            - List of RejectedFile records if study fails

    Filtering logic:
        1. Normalize sequences using orientation priority
        2. Check if study has required sequences (with allowed missing threshold)
    """
    rejected_files = []

    # Normalize sequences
    normalized_sequences, normalization_rejections = normalize_study_sequences(
        study_path,
        config.sequences,
        config.orientation_priority,
        dry_run,
    )
    rejected_files.extend(normalization_rejections)

    # Check how many required sequences are present
    present_sequences = normalized_sequences.intersection(set(config.sequences))
    missing_count = len(config.sequences) - len(present_sequences)

    if missing_count > config.allowed_missing_sequences_per_study:
        # Study fails filter
        missing_sequences = set(config.sequences) - present_sequences
        rejection_reason = (
            f"Study missing {missing_count} sequences "
            f"(allowed: {config.allowed_missing_sequences_per_study}). "
            f"Missing: {sorted(missing_sequences)}"
        )

        # Create rejection records for all files in the study
        for nrrd_file in study_path.glob("*.nrrd"):
            rejected_files.append(
                RejectedFile(
                    source_path=str(nrrd_file),
                    filename=nrrd_file.name,
                    patient_id=patient_id,
                    study_name=study_num,
                    rejection_reason=rejection_reason,
                    source_type="filtering_study",
                    stage=1,
                )
            )

        return False, rejected_files

    logger.debug(
        f"Study {patient_id}/{study_num} passed filter "
        f"({len(present_sequences)}/{len(config.sequences)} sequences present)"
    )
    return True, rejected_files


def filter_patient(
    patient_path: Path, config: FilteringConfig, dry_run: bool
) -> Tuple[bool, List[RejectedFile], int]:
    """Filter all studies for a patient and check minimum study requirement.

    Args:
        patient_path: Path to patient directory (e.g., .../P1/).
        config: Filtering configuration.
        dry_run: If True, simulate without making changes.

    Returns:
        Tuple of:
            - Boolean indicating if patient should be kept
            - List of RejectedFile records
            - Number of valid studies found

    Filtering logic:
        1. Filter each study individually
        2. Check if patient meets minimum studies requirement
        3. If not, reject entire patient
    """
    patient_id = patient_path.name  # e.g., 'P1'
    rejected_files = []
    valid_study_count = 0

    # Get all study directories (numeric directories)
    study_dirs = sorted(
        [d for d in patient_path.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    if not study_dirs:
        logger.warning(f"No study directories found for patient {patient_id}")
        return False, rejected_files, 0

    # Filter each study
    valid_studies = []
    for study_dir in study_dirs:
        study_num = study_dir.name
        passes_filter, study_rejections = filter_study(
            patient_id, study_num, study_dir, config, dry_run
        )

        if passes_filter:
            valid_studies.append(study_dir)
            valid_study_count += 1
        else:
            rejected_files.extend(study_rejections)
            # Delete failed study directory
            if not dry_run:
                logger.info(
                    f"Deleting study directory {patient_id}/{study_num} "
                    f"(failed sequence requirements)"
                )
                shutil.rmtree(study_dir)

    # Check minimum studies per patient
    if valid_study_count < config.min_studies_per_patient:
        rejection_reason = (
            f"Patient has only {valid_study_count} valid studies "
            f"(minimum required: {config.min_studies_per_patient})"
        )

        # Add rejection records for all remaining valid studies
        for study_dir in valid_studies:
            study_num = study_dir.name
            for nrrd_file in study_dir.glob("*.nrrd"):
                rejected_files.append(
                    RejectedFile(
                        source_path=str(nrrd_file),
                        filename=nrrd_file.name,
                        patient_id=patient_id,
                        study_name=study_num,
                        rejection_reason=rejection_reason,
                        source_type="filtering_patient",
                        stage=1,
                    )
                )

        # Delete entire patient directory
        if not dry_run:
            logger.info(
                f"Deleting patient directory {patient_id} "
                f"(only {valid_study_count} valid studies, "
                f"minimum required: {config.min_studies_per_patient})"
            )
            shutil.rmtree(patient_path)

        return False, rejected_files, valid_study_count

    logger.info(f"Patient {patient_id} passed filter ({valid_study_count} valid studies)")
    return True, rejected_files, valid_study_count


def append_to_rejected_files_csv(
    rejected_files: List[RejectedFile], csv_path: Path
) -> None:
    """Append new rejected files to existing rejected_files.csv.

    This function handles both cases:
    1. CSV exists: Read existing data, add stage column if missing, append new rejections
    2. CSV doesn't exist: Create new CSV with all rejections

    Args:
        rejected_files: New rejected file records to append.
        csv_path: Path to rejected_files.csv.

    Raises:
        OSError: If file operations fail.
    """
    if not rejected_files:
        logger.info("No rejected files to append to CSV")
        return

    # Ensure output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing CSV if it exists
    existing_rejections = []
    if csv_path.exists():
        logger.info(f"Reading existing rejected files from {csv_path}")
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Add stage field if missing (for backward compatibility)
                if "stage" not in row:
                    row["stage"] = 0
                existing_rejections.append(row)

    # Combine existing and new rejections
    logger.info(
        f"Appending {len(rejected_files)} new rejections to "
        f"{len(existing_rejections)} existing records"
    )

    # Write combined data
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "source_path",
            "filename",
            "patient_id",
            "study_name",
            "rejection_reason",
            "source_type",
            "stage",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Write existing rejections
        for row in existing_rejections:
            writer.writerow(row)

        # Write new rejections
        for rejected in rejected_files:
            writer.writerow(
                {
                    "source_path": rejected.source_path,
                    "filename": rejected.filename,
                    "patient_id": rejected.patient_id,
                    "study_name": rejected.study_name,
                    "rejection_reason": rejected.rejection_reason,
                    "source_type": rejected.source_type,
                    "stage": rejected.stage,
                }
            )

    logger.info(f"Updated rejected files CSV: {csv_path}")


def remove_non_required_sequences(
    mengrowth_dir: Path,
    required_sequences: List[str],
    dry_run: bool,
) -> int:
    """Remove all sequences that are not in the required sequences list.

    Args:
        mengrowth_dir: Path to MenGrowth-2025 directory.
        required_sequences: List of sequences to keep (e.g., ['t1c', 't1n', 't2f', 't2w']).
        dry_run: If True, simulate without actually deleting files.

    Returns:
        Number of files deleted.

    Examples:
        If required_sequences=['t1c', 't1n', 't2f', 't2w'] and a study has
        ['t1c.nrrd', 't1n.nrrd', 't2f.nrrd', 't2w.nrrd', 'dwi.nrrd', 'swi.nrrd'],
        then 'dwi.nrrd' and 'swi.nrrd' will be deleted.
    """
    deleted_count = 0
    required_set = set(required_sequences)

    patient_dirs = sorted([d for d in mengrowth_dir.iterdir() if d.is_dir()])
    total_patients = len(patient_dirs)
    logger.info(
        f"Removing non-required sequences from {total_patients} patients "
        f"(keeping: {required_sequences})..."
    )

    for idx, patient_dir in enumerate(patient_dirs, 1):
        patient_deleted = 0

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            for nrrd_file in study_dir.glob("*.nrrd"):
                sequence_name = nrrd_file.stem

                if sequence_name not in required_set:
                    logger.debug(
                        f"Deleting non-required sequence: "
                        f"{patient_dir.name}/{study_dir.name}/{nrrd_file.name}"
                    )

                    if not dry_run:
                        nrrd_file.unlink()

                    deleted_count += 1
                    patient_deleted += 1

        if patient_deleted > 0:
            logger.debug(
                f"Patient {patient_dir.name}: removed {patient_deleted} non-required files"
            )

        if idx % 25 == 0 or idx == total_patients:
            logger.info(
                f"  Sequence cleanup progress: {idx}/{total_patients} patients "
                f"({deleted_count} files removed so far)"
            )

    logger.info(f"Removed {deleted_count} non-required sequence files")
    return deleted_count


def reid_patients_and_studies(
    mengrowth_dir: Path,
    dry_run: bool,
) -> Dict[str, Dict[str, any]]:
    """Rename patients to MenGrowth-XXXX format and create id_mapping.json.

    Args:
        mengrowth_dir: Path to MenGrowth-2025 directory.
        dry_run: If True, simulate without actually renaming directories.

    Returns:
        ID mapping dictionary with structure:
        {
            "P1": {
                "new_id": "MenGrowth-0001",
                "studies": {
                    "0": "MenGrowth-0001-000",
                    "1": "MenGrowth-0001-001"
                }
            },
            ...
        }

    Examples:
        P1 -> MenGrowth-0001 (with studies 0, 1, 2)
        P42 -> MenGrowth-0002 (with studies 0, 1)
        Creates MenGrowth-0001-000, MenGrowth-0001-001, MenGrowth-0001-002, etc.
    """
    logger.info("Re-identifying patients and studies...")

    # Get all patient directories and sort them
    patient_dirs = sorted(
        [d for d in mengrowth_dir.iterdir() if d.is_dir() and d.name.startswith("P")],
        key=lambda p: int(p.name[1:]),  # Sort by numeric part after 'P'
    )

    id_mapping = {}
    new_patient_counter = 1
    total_patients = len(patient_dirs)

    for patient_dir in patient_dirs:
        old_patient_id = patient_dir.name  # e.g., 'P1', 'P42'
        new_patient_id = f"MenGrowth-{new_patient_counter:04d}"  # e.g., 'MenGrowth-0001'

        logger.info(
            f"Re-identifying {new_patient_counter}/{total_patients}: "
            f"{old_patient_id} -> {new_patient_id}"
        )

        # Get all study directories and sort them numerically
        study_dirs = sorted(
            [d for d in patient_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda s: int(s.name),
        )

        # Create study mappings
        study_mappings = {}
        for study_idx, study_dir in enumerate(study_dirs):
            old_study_id = study_dir.name  # e.g., '0', '1', '2'
            new_study_id = f"{new_patient_id}-{study_idx:03d}"  # e.g., 'MenGrowth-0001-000'

            study_mappings[old_study_id] = new_study_id

            logger.debug(
                f"  Study: {old_patient_id}/{old_study_id} -> {new_study_id}"
            )

            # Rename study directory
            if not dry_run:
                new_study_path = patient_dir / new_study_id
                study_dir.rename(new_study_path)

        # Add to mapping dictionary
        id_mapping[old_patient_id] = {
            "new_id": new_patient_id,
            "studies": study_mappings,
        }

        # Rename patient directory
        if not dry_run:
            new_patient_path = mengrowth_dir / new_patient_id
            patient_dir.rename(new_patient_path)

        new_patient_counter += 1

    logger.info(f"Re-identified {len(id_mapping)} patients")

    # Write id_mapping.json
    mapping_file = mengrowth_dir.parent / "id_mapping.json"

    if not dry_run:
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(id_mapping, f, indent=2, ensure_ascii=False)

        logger.info(f"ID mapping written to: {mapping_file}")

    return id_mapping


def filter_raw_data(
    data_root: Path,
    config: PreprocessingConfig,
    metadata_manager: Optional["MetadataManager"] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Filter reorganized raw data based on quality and completeness criteria.

    This is the main entry point for the filtering pipeline. It operates on
    reorganized data and applies filtering criteria to ensure data quality.

    Args:
        data_root: Root directory containing reorganized data
            (should contain MenGrowth-2025/ subdirectory).
        config: Preprocessing configuration with filtering parameters.
        metadata_manager: Optional metadata manager to track patient exclusions
            and apply ID mapping.
        dry_run: If True, simulate filtering without making changes.

    Returns:
        Statistics dictionary with keys:
            - 'patients_kept': Number of patients retained
            - 'patients_removed': Number of patients deleted
            - 'studies_processed': Total studies processed
            - 'files_renamed': Number of files renamed for normalization
            - 'sequences_removed': Number of non-required sequences deleted (if enabled)
            - 'patients_renamed': Number of patients re-identified (if enabled)

    Raises:
        ValueError: If filtering configuration is missing.
        FileNotFoundError: If data_root doesn't exist.

    Examples:
        >>> stats = filter_raw_data(Path('/data'), config, dry_run=True)
        >>> print(f"Kept {stats['patients_kept']} patients")
    """
    if config.filtering is None:
        raise ValueError(
            "Filtering configuration is missing. "
            "Please add 'filtering:' section to your config YAML."
        )

    filtering_config = config.filtering

    # Validate data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

    mengrowth_dir = data_root / "MenGrowth-2025"
    if not mengrowth_dir.exists():
        raise FileNotFoundError(
            f"MenGrowth-2025 directory not found in {data_root}. "
            "Please run reorganization first."
        )

    logger.info("=" * 80)
    logger.info("Starting data filtering")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Required sequences: {filtering_config.sequences}")
    logger.info(
        f"Allowed missing sequences per study: "
        f"{filtering_config.allowed_missing_sequences_per_study}"
    )
    logger.info(
        f"Minimum studies per patient: {filtering_config.min_studies_per_patient}"
    )
    logger.info(f"Orientation priority: {filtering_config.orientation_priority}")
    logger.info("=" * 80)

    # Initialize statistics
    stats = {
        "patients_kept": 0,
        "patients_removed": 0,
        "studies_processed": 0,
        "files_renamed": 0,
    }

    all_rejected_files = []

    # Get all patient directories
    patient_dirs = sorted(
        [
            d
            for d in mengrowth_dir.iterdir()
            if d.is_dir() and d.name.startswith("P")
        ]
    )

    total_patients = len(patient_dirs)
    logger.info(f"Found {total_patients} patient directories to process")

    # Process each patient
    for idx, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name

        # Count studies before filtering
        study_count = len([d for d in patient_dir.iterdir() if d.is_dir()])
        stats["studies_processed"] += study_count

        logger.info(
            f"Filtering patient {idx}/{total_patients}: {patient_id} "
            f"({study_count} studies)"
        )

        # Filter patient
        keep_patient, patient_rejections, valid_studies = filter_patient(
            patient_dir, filtering_config, dry_run
        )

        all_rejected_files.extend(patient_rejections)

        if keep_patient:
            stats["patients_kept"] += 1
            # Mark as included in metadata
            if metadata_manager:
                metadata_manager.mark_included(patient_id)
        else:
            stats["patients_removed"] += 1
            # Track exclusion in metadata
            if metadata_manager:
                # Determine the exclusion reason from rejections
                if patient_rejections:
                    reason = patient_rejections[0].rejection_reason
                else:
                    reason = "filtered_out"
                metadata_manager.mark_excluded(patient_id, reason)

    # Reconcile metadata: exclude patients that have no data directory
    if metadata_manager:
        existing_patient_ids = {d.name for d in mengrowth_dir.iterdir() if d.is_dir()}
        for patient_id in metadata_manager.get_patient_ids():
            patient = metadata_manager.get_patient(patient_id)
            if patient and patient.included and patient_id not in existing_patient_ids:
                metadata_manager.mark_excluded(patient_id, "no_data_directory")
                logger.info(f"Phantom patient fix: {patient_id} excluded (no data directory)")

    # Append rejections to CSV
    rejected_csv_path = data_root / "rejected_files.csv"
    append_to_rejected_files_csv(all_rejected_files, rejected_csv_path)

    # Remove non-required sequences if configured
    if filtering_config.keep_only_required_sequences:
        logger.info("=" * 80)
        sequences_removed = remove_non_required_sequences(
            mengrowth_dir, filtering_config.sequences, dry_run
        )
        stats["sequences_removed"] = sequences_removed
        logger.info("=" * 80)
    else:
        stats["sequences_removed"] = 0

    # Re-identify patients and studies if configured
    if filtering_config.reid_patients:
        logger.info("=" * 80)
        id_mapping = reid_patients_and_studies(mengrowth_dir, dry_run)
        stats["patients_renamed"] = len(id_mapping)

        # Apply ID mapping to metadata
        if metadata_manager and id_mapping:
            # Convert id_mapping format to what MetadataManager expects
            # From: {"P1": {"new_id": "MenGrowth-0001", "studies": {...}}, ...}
            # To: {"MenGrowth-0001": {"original_id": "P1", ...}, ...}
            metadata_mapping = {}
            for original_id, mapping_info in id_mapping.items():
                new_id = mapping_info["new_id"]
                metadata_mapping[new_id] = {
                    "original_id": original_id,
                    "studies": mapping_info.get("studies", {}),
                }
            metadata_manager.apply_id_mapping(metadata_mapping)
            logger.info(f"Applied ID mapping to metadata for {len(metadata_mapping)} patients")

        logger.info("=" * 80)
    else:
        stats["patients_renamed"] = 0

    # Log final statistics
    logger.info("=" * 80)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Patients kept: {stats['patients_kept']}")
    logger.info(f"Patients removed: {stats['patients_removed']}")
    logger.info(f"Studies processed: {stats['studies_processed']}")
    logger.info(f"Total rejections logged: {len(all_rejected_files)}")

    if filtering_config.keep_only_required_sequences:
        logger.info(f"Non-required sequences removed: {stats['sequences_removed']}")

    if filtering_config.reid_patients:
        logger.info(f"Patients renamed: {stats['patients_renamed']}")
        logger.info(f"ID mapping saved to: {data_root}/id_mapping.json")

    logger.info("=" * 80)

    return stats
