"""Study discovery, validation, and BraTS-format input preparation.

This module handles the first stage of the segmentation pipeline:
discovering preprocessed studies, validating their completeness and shape,
correcting shapes if needed, and creating BraTS-compliant symlink directories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import tempfile

import nibabel as nib
import numpy as np

from mengrowth.segmentation.config import SegmentationConfig

logger = logging.getLogger(__name__)


@dataclass
class StudyInfo:
    """Information about a discovered study.

    Attributes:
        patient_id: Patient identifier (e.g., MenGrowth-0001).
        study_id: Study identifier (e.g., MenGrowth-0001-000).
        study_path: Absolute path to the study directory.
        available_modalities: Modalities found in the study directory.
        shape_per_modality: Mapping of modality to volume shape.
        is_complete: Whether all required modalities are present.
        shape_issues: List of shape mismatch descriptions.
    """

    patient_id: str = ""
    study_id: str = ""
    study_path: Path = field(default_factory=Path)
    available_modalities: List[str] = field(default_factory=list)
    shape_per_modality: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    is_complete: bool = False
    shape_issues: List[str] = field(default_factory=list)


def discover_studies(
    input_root: str | Path,
    config: SegmentationConfig,
    patient_filter: Optional[str] = None,
) -> List[StudyInfo]:
    """Discover preprocessed studies under input_root.

    Walks MenGrowth-XXXX/MenGrowth-XXXX-YYY/ directories, reads NIfTI
    headers to collect shapes, and flags completeness.

    Args:
        input_root: Root directory of preprocessed data.
        config: Segmentation configuration.
        patient_filter: If set, only discover studies for this patient ID.

    Returns:
        Sorted list of StudyInfo objects.
    """
    input_root = Path(input_root)
    studies: List[StudyInfo] = []

    patient_dirs = sorted(
        d
        for d in input_root.iterdir()
        if d.is_dir()
        and d.name.startswith("MenGrowth-")
        and not d.name.startswith("_temp_")
    )

    if patient_filter:
        patient_dirs = [d for d in patient_dirs if d.name == patient_filter]
        if not patient_dirs:
            logger.warning(f"Patient {patient_filter} not found in {input_root}")

    for patient_dir in patient_dirs:
        study_dirs = sorted(
            d
            for d in patient_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_temp_")
        )

        for study_dir in study_dirs:
            info = StudyInfo(
                patient_id=patient_dir.name,
                study_id=study_dir.name,
                study_path=study_dir,
            )

            # Find available modalities
            for mod in config.modalities:
                nifti_path = study_dir / f"{mod}.nii.gz"
                if nifti_path.exists():
                    info.available_modalities.append(mod)
                    try:
                        img = nib.load(str(nifti_path))
                        info.shape_per_modality[mod] = img.shape[:3]
                    except Exception as e:
                        logger.warning(f"Failed to read header for {nifti_path}: {e}")
                        info.shape_issues.append(f"{mod}: failed to read header")

            info.is_complete = set(config.modalities).issubset(
                set(info.available_modalities)
            )

            studies.append(info)

    logger.info(
        f"Discovered {len(studies)} studies "
        f"({sum(1 for s in studies if s.is_complete)} complete)"
    )
    return studies


def validate_study(
    study_info: StudyInfo, config: SegmentationConfig
) -> Tuple[bool, List[str]]:
    """Validate a study for segmentation readiness.

    Checks that all required modalities are present and that shapes
    are within tolerance of the expected shape.

    Args:
        study_info: Study information from discovery.
        config: Segmentation configuration.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues: List[str] = []

    if not study_info.is_complete:
        missing = set(config.modalities) - set(study_info.available_modalities)
        issues.append(f"Missing modalities: {sorted(missing)}")

    expected = config.expected_shape
    tol = config.shape_tolerance

    for mod, shape in study_info.shape_per_modality.items():
        for axis, (actual, exp) in enumerate(zip(shape, expected)):
            if abs(actual - exp) > tol:
                issues.append(
                    f"{mod} shape axis {axis}: {actual} vs expected {exp} "
                    f"(tolerance {tol})"
                )

    issues.extend(study_info.shape_issues)
    is_valid = len(issues) == 0
    return is_valid, issues


def correct_shape(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int],
) -> Dict:
    """Pad or crop a NIfTI volume to the exact target shape.

    Padding uses zeros. Cropping is center-symmetric. The affine origin
    is adjusted to account for added/removed voxels.

    Args:
        input_path: Path to input NIfTI file.
        output_path: Path to write corrected NIfTI file.
        target_shape: Desired (x, y, z) shape.

    Returns:
        Dict with correction metadata (original_shape, pad_before, crop_before).
    """
    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj)
    original_shape = data.shape[:3]
    affine = img.affine.copy()

    pad_before = [0, 0, 0]
    pad_after = [0, 0, 0]
    crop_before = [0, 0, 0]
    crop_after = [0, 0, 0]

    for axis in range(3):
        diff = target_shape[axis] - original_shape[axis]
        if diff > 0:
            # Pad: split evenly, extra voxel on the after side
            pad_before[axis] = diff // 2
            pad_after[axis] = diff - pad_before[axis]
        elif diff < 0:
            # Crop: center crop
            crop_before[axis] = (-diff) // 2
            crop_after[axis] = (-diff) - crop_before[axis]

    # Apply cropping
    slices = tuple(
        slice(crop_before[ax], original_shape[ax] - crop_after[ax]) for ax in range(3)
    )
    data = data[slices]

    # Apply padding
    pad_widths = [(pad_before[ax], pad_after[ax]) for ax in range(3)]
    # Handle 4D data (unlikely but safe)
    if len(data.shape) > 3:
        pad_widths.extend([(0, 0)] * (len(data.shape) - 3))
    data = np.pad(data, pad_widths, mode="constant", constant_values=0)

    # Adjust affine origin for the net shift
    voxel_size = np.abs(np.diag(affine[:3, :3]))
    net_shift = np.array(
        [(crop_before[ax] - pad_before[ax]) for ax in range(3)], dtype=float
    )
    affine[:3, 3] += affine[:3, :3] @ net_shift

    corrected_img = nib.Nifti1Image(data, affine, img.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(corrected_img, str(output_path))

    return {
        "original_shape": list(original_shape),
        "target_shape": list(target_shape),
        "pad_before": pad_before,
        "crop_before": crop_before,
    }


def prepare_brats_input(
    studies: List[StudyInfo],
    config: SegmentationConfig,
) -> Tuple[Path, Dict]:
    """Create BraTS-compliant input directory with symlinks.

    Creates a work directory with BraTS-named symlinks (or corrected copies
    for shape mismatches) and a name mapping JSON.

    Args:
        studies: List of validated StudyInfo objects to include.
        config: Segmentation configuration.

    Returns:
        Tuple of (work_dir, name_mapping_dict).
    """
    # Determine work directory
    if config.work_dir:
        work_dir = Path(config.work_dir)
    else:
        base = Path(config.log_dir) if config.log_dir else Path(tempfile.gettempdir())
        work_dir = Path(tempfile.mkdtemp(prefix="brats_men_", dir=str(base)))

    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_map: Dict[str, Dict] = {}
    expected = config.expected_shape

    for idx, study in enumerate(studies):
        brats_name = config.brats_name_schema.format(idx)
        brats_subj_dir = input_dir / brats_name
        brats_subj_dir.mkdir(parents=True, exist_ok=True)

        shape_corrected = False
        original_shape_info: Dict[str, List[int]] = {}

        for mod in config.modalities:
            if mod not in study.available_modalities:
                continue

            src_path = study.study_path / f"{mod}.nii.gz"
            link_name = f"{brats_name}-{mod}.nii.gz"
            dst_path = brats_subj_dir / link_name

            # Check if shape correction is needed
            shape = study.shape_per_modality.get(mod, ())
            needs_correction = len(shape) == 3 and any(
                shape[ax] != expected[ax] for ax in range(3)
            )

            if needs_correction:
                logger.info(
                    f"  Shape correction {mod}: {shape} -> {expected} "
                    f"for {study.study_id}"
                )
                correction_info = correct_shape(src_path, dst_path, expected)
                shape_corrected = True
                original_shape_info[mod] = correction_info["original_shape"]
            else:
                os.symlink(src_path.resolve(), dst_path)

        name_map[brats_name] = {
            "patient_id": study.patient_id,
            "study_id": study.study_id,
            "study_path": str(study.study_path),
            "shape_corrected": shape_corrected,
            "original_shape": original_shape_info if shape_corrected else {},
        }

        logger.info(
            f"  [{idx + 1:3d}/{len(studies)}] {study.study_id} -> {brats_name}"
            + (" (shape corrected)" if shape_corrected else "")
        )

    # Save name mapping
    map_file = work_dir / "name_map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2)

    logger.info(f"Name mapping saved to: {map_file}")
    logger.info(f"BraTS input prepared at: {input_dir}")
    logger.info(f"Prepared {len(name_map)} studies for segmentation")

    # Print work_dir as last stdout line for shell capture
    print(f"WORK_DIR={work_dir}")

    return work_dir, name_map
