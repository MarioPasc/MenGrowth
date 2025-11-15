"""Reorganize raw MRI data into standardized directory structure.

This module provides utilities to reorganize multimodal MRI data from various
input structures into a standardized format for reproducible research pipelines.

The reorganization follows the structure:
    {output_root}/MenGrowth-2025/P{patient_id}/{study_number}/*.nrrd

Input sources:
    - source/baseline/RM/{modality}/{patient}/files.nrrd
    - source/baseline/TC/{patient}/files.nrrd
    - source/controls/{patient}/{control1,control2,...}/files.nrrd
    - extension_1/{patient_id}/{primera,segunda,...}/files.nrrd
"""

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from mengrowth.preprocessing.config import RawDataConfig

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class RejectedFile:
    """Record of a file that was rejected during reorganization.

    Attributes:
        source_path: Original file path.
        filename: Name of the file.
        patient_id: Extracted or intended patient ID (if applicable).
        study_name: Original study directory name (if applicable).
        rejection_reason: Explanation for why the file was rejected.
        source_type: Type of source (baseline_RM, baseline_TC, controls, extension).
    """

    source_path: str
    filename: str
    patient_id: str
    study_name: str
    rejection_reason: str
    source_type: str


def should_exclude_file(filepath: Path, exclusion_patterns: List[str]) -> Tuple[bool, str]:
    """Check if file should be excluded based on exclusion patterns.

    Args:
        filepath: Path to the file to check.
        exclusion_patterns: List of glob patterns to exclude.

    Returns:
        Tuple of (should_exclude, reason) where reason is the matched pattern or empty string.

    Examples:
        >>> should_exclude_file(Path('TC_P1_seg.nrrd'), ['*seg*.nrrd'])
        (True, 'Matched exclusion pattern: *seg*.nrrd')
        >>> should_exclude_file(Path('FLAIR_P1.nrrd'), ['*seg*.nrrd'])
        (False, '')
    """
    filename = filepath.name
    for pattern in exclusion_patterns:
        if Path(filename).match(pattern):
            return True, f"Matched exclusion pattern: {pattern}"
    return False, ""


def extract_patient_id(patient_str: str) -> str:
    """Extract and normalize patient ID, ensuring 'P' prefix.

    Args:
        patient_str: Patient identifier (e.g., 'P1', '85', 'P42').

    Returns:
        Normalized patient ID with 'P' prefix (e.g., 'P1', 'P85', 'P42').

    Examples:
        >>> extract_patient_id('P1')
        'P1'
        >>> extract_patient_id('85')
        'P85'
        >>> extract_patient_id('042')
        'P42'
    """
    patient_str = patient_str.strip()

    # If already has P prefix, return as-is
    if patient_str.upper().startswith('P'):
        # Remove any leading zeros after P
        numeric_part = patient_str[1:].lstrip('0') or '0'
        return f"P{numeric_part}"

    # If numeric, add P prefix and remove leading zeros
    numeric_part = patient_str.lstrip('0') or '0'
    return f"P{numeric_part}"


def scan_source_baseline(
    baseline_path: Path, config: RawDataConfig, rejected_files: List[RejectedFile]
) -> List[Tuple[Path, str, str]]:
    """Scan source/baseline directory for RM and TC data.

    Both RM (with modality subdirectories) and TC are merged into study '0'.

    Args:
        baseline_path: Path to source/baseline directory.
        config: Raw data configuration with exclusion patterns.
        rejected_files: List to append rejected file records to.

    Returns:
        List of tuples: (source_file_path, patient_id, study_number).

    Examples:
        >>> rejected = []
        >>> files = scan_source_baseline(Path('data/source/baseline'), config, rejected)
        >>> files[0]
        (Path('.../RM/FLAIR/P1/FLAIR_P1.nrrd'), 'P1', '0')
    """
    file_list: List[Tuple[Path, str, str]] = []

    if not baseline_path.exists():
        logger.warning(f"Baseline path does not exist: {baseline_path}")
        return file_list

    # Process RM directory (has modality subdirectories)
    rm_path = baseline_path / "RM"
    if rm_path.exists():
        logger.info(f"Scanning RM directory: {rm_path}")
        for modality_dir in rm_path.iterdir():
            if not modality_dir.is_dir():
                continue

            for patient_dir in modality_dir.iterdir():
                if not patient_dir.is_dir():
                    continue

                patient_id = extract_patient_id(patient_dir.name)

                for file_path in patient_dir.glob("*.nrrd"):
                    should_exclude, reason = should_exclude_file(file_path, config.exclusion_patterns)
                    if should_exclude:
                        logger.debug(f"Excluding file: {file_path} - {reason}")
                        rejected_files.append(
                            RejectedFile(
                                source_path=str(file_path),
                                filename=file_path.name,
                                patient_id=patient_id,
                                study_name="baseline",
                                rejection_reason=reason,
                                source_type="baseline_RM",
                            )
                        )
                        continue

                    file_list.append((file_path, patient_id, "0"))
                    logger.debug(f"Found RM file: {file_path} -> P{patient_id}/0")

    # Process TC directory (patient folders directly)
    tc_path = baseline_path / "TC"
    if tc_path.exists():
        logger.info(f"Scanning TC directory: {tc_path}")
        for patient_dir in tc_path.iterdir():
            if not patient_dir.is_dir():
                continue

            patient_id = extract_patient_id(patient_dir.name)

            for file_path in patient_dir.glob("*.nrrd"):
                should_exclude, reason = should_exclude_file(file_path, config.exclusion_patterns)
                if should_exclude:
                    logger.debug(f"Excluding file: {file_path} - {reason}")
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name="baseline",
                            rejection_reason=reason,
                            source_type="baseline_TC",
                        )
                    )
                    continue

                file_list.append((file_path, patient_id, "0"))
                logger.debug(f"Found TC file: {file_path} -> P{patient_id}/0")

    logger.info(f"Found {len(file_list)} files in baseline (study 0)")
    return file_list


def scan_source_controls(
    controls_path: Path, config: RawDataConfig, rejected_files: List[RejectedFile]
) -> List[Tuple[Path, str, str]]:
    """Scan source/controls directory for follow-up studies.

    Control directories (control1, control2, ...) map to study numbers (1, 2, ...).

    Args:
        controls_path: Path to source/controls directory.
        config: Raw data configuration with study mappings and exclusion patterns.
        rejected_files: List to append rejected file records to.

    Returns:
        List of tuples: (source_file_path, patient_id, study_number).

    Examples:
        >>> rejected = []
        >>> files = scan_source_controls(Path('data/source/controls'), config, rejected)
        >>> files[0]
        (Path('.../P40/control1/T1_P40_01.nrrd'), 'P40', '1')
    """
    file_list: List[Tuple[Path, str, str]] = []

    if not controls_path.exists():
        logger.warning(f"Controls path does not exist: {controls_path}")
        return file_list

    logger.info(f"Scanning controls directory: {controls_path}")

    for patient_dir in controls_path.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = extract_patient_id(patient_dir.name)

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            # Map control directory name to study number
            study_name = study_dir.name.lower()
            try:
                study_number = config.get_study_number(study_name)
            except KeyError:
                logger.warning(
                    f"Unknown study directory '{study_dir.name}' for patient {patient_id}, skipping"
                )
                # Track files in unknown study directories as rejected
                for file_path in study_dir.glob("*.nrrd"):
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name=study_dir.name,
                            rejection_reason=f"Unknown study directory name '{study_dir.name}'",
                            source_type="controls",
                        )
                    )
                continue

            for file_path in study_dir.glob("*.nrrd"):
                should_exclude, reason = should_exclude_file(file_path, config.exclusion_patterns)
                if should_exclude:
                    logger.debug(f"Excluding file: {file_path} - {reason}")
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name=study_name,
                            rejection_reason=reason,
                            source_type="controls",
                        )
                    )
                    continue

                file_list.append((file_path, patient_id, study_number))
                logger.debug(f"Found control file: {file_path} -> P{patient_id}/{study_number}")

    logger.info(f"Found {len(file_list)} files in controls")
    return file_list


def scan_extension(
    extension_path: Path, config: RawDataConfig, rejected_files: List[RejectedFile]
) -> List[Tuple[Path, str, str]]:
    """Scan extension_1 directory for additional patient cohort.

    Patient IDs are converted to 'P' prefix format (e.g., 85 -> P85).
    Study directories (primera, segunda, ...) map to study numbers (0, 1, ...).

    Args:
        extension_path: Path to extension_1 directory.
        config: Raw data configuration with study mappings and exclusion patterns.
        rejected_files: List to append rejected file records to.

    Returns:
        List of tuples: (source_file_path, patient_id, study_number).

    Examples:
        >>> rejected = []
        >>> files = scan_extension(Path('data/extension_1'), config, rejected)
        >>> files[0]
        (Path('.../85/primera/FLAIR.nrrd'), 'P85', '0')
    """
    file_list: List[Tuple[Path, str, str]] = []

    if not extension_path.exists():
        logger.warning(f"Extension path does not exist: {extension_path}")
        return file_list

    logger.info(f"Scanning extension directory: {extension_path}")

    for patient_dir in extension_path.iterdir():
        if not patient_dir.is_dir():
            continue

        # Convert numeric ID to P-prefixed format
        patient_id = extract_patient_id(patient_dir.name)

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            # Map study directory name to study number
            study_name = study_dir.name.lower()
            try:
                study_number = config.get_study_number(study_name)
            except KeyError:
                logger.warning(
                    f"Unknown study directory '{study_dir.name}' for patient {patient_id}, skipping"
                )
                # Track files in unknown study directories as rejected
                for file_path in study_dir.glob("*.nrrd"):
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name=study_dir.name,
                            rejection_reason=f"Unknown study directory name '{study_dir.name}'",
                            source_type="extension",
                        )
                    )
                continue

            for file_path in study_dir.glob("*.nrrd"):
                should_exclude, reason = should_exclude_file(file_path, config.exclusion_patterns)
                if should_exclude:
                    logger.debug(f"Excluding file: {file_path} - {reason}")
                    rejected_files.append(
                        RejectedFile(
                            source_path=str(file_path),
                            filename=file_path.name,
                            patient_id=patient_id,
                            study_name=study_name,
                            rejection_reason=reason,
                            source_type="extension",
                        )
                    )
                    continue

                file_list.append((file_path, patient_id, study_number))
                logger.debug(f"Found extension file: {file_path} -> P{patient_id}/{study_number}")

    logger.info(f"Found {len(file_list)} files in extension")
    return file_list


def copy_and_organize(
    file_list: List[Tuple[Path, str, str]],
    output_root: Path,
    config: RawDataConfig,
    rejected_files: List[RejectedFile],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy and organize files into standardized structure with filename normalization.

    Args:
        file_list: List of (source_path, patient_id, study_number) tuples.
        output_root: Root directory for organized output.
        config: Raw data configuration with modality synonyms and output structure.
        rejected_files: List to append rejected file records to (for duplicates).
        dry_run: If True, log operations without actually copying files.

    Returns:
        Dictionary with statistics: {'copied': N, 'skipped': N, 'errors': N}.

    Raises:
        OSError: If directory creation or file copying fails.
    """
    stats = {"copied": 0, "skipped": 0, "errors": 0}

    # Track processed files to avoid duplicates
    processed: Set[Tuple[str, str, str]] = set()

    for source_path, patient_id, study_number in file_list:
        # Standardize modality name from filename
        standardized_modality, matched = config.standardize_modality(source_path.name)
        standardized_filename = f"{standardized_modality}.nrrd"

        # Track files that didn't match any modality pattern
        if not matched:
            logger.warning(f"No modality pattern matched for: {source_path}")
            rejected_files.append(
                RejectedFile(
                    source_path=str(source_path),
                    filename=source_path.name,
                    patient_id=patient_id,
                    study_name=study_number,
                    rejection_reason=f"No modality pattern matched (kept as '{standardized_modality}.nrrd')",
                    source_type="unmatched_pattern",
                )
            )

        # Create unique key to detect duplicates
        file_key = (patient_id, study_number, standardized_filename)

        if file_key in processed:
            reason = f"Duplicate file for {patient_id}/study{study_number}/{standardized_filename}"
            logger.warning(f"Duplicate file detected: {source_path} - {reason}")
            rejected_files.append(
                RejectedFile(
                    source_path=str(source_path),
                    filename=source_path.name,
                    patient_id=patient_id,
                    study_name=study_number,
                    rejection_reason=reason,
                    source_type="duplicate",
                )
            )
            stats["skipped"] += 1
            continue

        # Build output path (patient_id already has 'P' prefix)
        output_dir = output_root / "MenGrowth-2025" / patient_id / study_number
        output_path = output_dir / standardized_filename

        # Log operation
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Copy: {source_path} -> {output_path}"
        )

        if not dry_run:
            try:
                # Create output directory if needed
                output_dir.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(source_path, output_path)
                stats["copied"] += 1
                processed.add(file_key)

            except Exception as e:
                error_msg = f"Failed to copy: {e}"
                logger.error(f"{error_msg} - {source_path} to {output_path}")
                rejected_files.append(
                    RejectedFile(
                        source_path=str(source_path),
                        filename=source_path.name,
                        patient_id=patient_id,
                        study_name=study_number,
                        rejection_reason=error_msg,
                        source_type="copy_error",
                    )
                )
                stats["errors"] += 1
        else:
            stats["copied"] += 1
            processed.add(file_key)

    return stats


def write_rejected_files_csv(
    rejected_files: List[RejectedFile], output_path: Path
) -> None:
    """Write rejected files report to CSV file.

    Args:
        rejected_files: List of rejected file records.
        output_path: Path where CSV file should be written.

    Raises:
        OSError: If file writing fails.
    """
    if not rejected_files:
        logger.info("No rejected files to write to CSV")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(rejected_files)} rejected files to {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "source_path",
            "filename",
            "patient_id",
            "study_name",
            "rejection_reason",
            "source_type",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for rejected in rejected_files:
            writer.writerow(
                {
                    "source_path": rejected.source_path,
                    "filename": rejected.filename,
                    "patient_id": rejected.patient_id,
                    "study_name": rejected.study_name,
                    "rejection_reason": rejected.rejection_reason,
                    "source_type": rejected.source_type,
                }
            )

    logger.info(f"Rejected files CSV written successfully to {output_path}")


def reorganize_raw_data(
    input_root: Path,
    output_root: Path,
    config: RawDataConfig,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Reorganize raw MRI data from complex nested structure to standardized format.

    This is the main orchestration function that:
    1. Scans all input sources (baseline, controls, extension)
    2. Collects file metadata (patient ID, study number)
    3. Copies files to standardized output structure with normalized names
    4. Excludes unwanted files (segmentations, transforms, etc.)
    5. Generates CSV report of all rejected files

    Args:
        input_root: Root directory containing 'source' and 'extension_1' folders.
        output_root: Root directory for reorganized output.
        config: Raw data configuration with mappings and exclusions.
        dry_run: If True, simulate operations without copying files.

    Returns:
        Dictionary with reorganization statistics.

    Raises:
        FileNotFoundError: If input_root does not exist.
        ValueError: If no files are found to process.

    Examples:
        >>> from pathlib import Path
        >>> from mengrowth.preprocessing.config import load_preprocessing_config
        >>> config = load_preprocessing_config(Path('configs/preprocessing.yaml'))
        >>> stats = reorganize_raw_data(
        ...     Path('/data/raw/processed'),
        ...     Path('/data/organized'),
        ...     config.raw_data,
        ...     dry_run=True
        ... )
        >>> print(f"Copied {stats['copied']} files")
    """
    if not input_root.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {input_root}")

    logger.info(f"Starting data reorganization from {input_root} to {output_root}")
    logger.info(f"Dry run mode: {dry_run}")

    # Track all rejected files
    rejected_files: List[RejectedFile] = []

    # Collect all files from different sources
    all_files = []

    # 1. Scan source/baseline (RM + TC -> study 0)
    baseline_path = input_root / "source" / "baseline"
    all_files.extend(scan_source_baseline(baseline_path, config, rejected_files))

    # 2. Scan source/controls (control1 -> study 1, control2 -> study 2, ...)
    controls_path = input_root / "source" / "controls"
    all_files.extend(scan_source_controls(controls_path, config, rejected_files))

    # 3. Scan extension_1 (primera -> study 0, segunda -> study 1, ...)
    extension_path = input_root / "extension_1"
    all_files.extend(scan_extension(extension_path, config, rejected_files))

    if not all_files:
        raise ValueError(
            f"No NRRD files found in {input_root}. Check input directory structure."
        )

    logger.info(f"Total files collected: {len(all_files)}")
    logger.info(f"Total files rejected during scanning: {len(rejected_files)}")

    # Copy and organize files
    stats = copy_and_organize(all_files, output_root, config, rejected_files, dry_run=dry_run)

    logger.info(
        f"Reorganization complete: {stats['copied']} copied, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    logger.info(f"Total files rejected: {len(rejected_files)}")

    # Write rejected files CSV report
    rejected_csv_path = output_root / "rejected_files.csv"
    write_rejected_files_csv(rejected_files, rejected_csv_path)

    return stats
