"""Quality filtering module for MenGrowth data curation.

This module provides validation functions for filtering data based on quality
metrics before preprocessing. All thresholds are configurable via YAML.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nrrd
import numpy as np

from mengrowth.preprocessing.config import QualityFilteringConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    passed: bool
    check_name: str
    message: str
    action: str = "warn"  # "warn" or "block"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileValidationReport:
    """Complete validation report for a single file."""

    file_path: Path
    patient_id: str
    study_id: str
    modality: str
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        """Check if any blocking issues were found."""
        return any(not r.passed and r.action == "block" for r in self.results)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(not r.passed and r.action == "warn" for r in self.results)

    @property
    def blocking_issues(self) -> List[ValidationResult]:
        """Get list of blocking issues."""
        return [r for r in self.results if not r.passed and r.action == "block"]

    @property
    def warnings(self) -> List[ValidationResult]:
        """Get list of warnings."""
        return [r for r in self.results if not r.passed and r.action == "warn"]


@dataclass
class StudyValidationReport:
    """Validation report for a study (all modalities)."""

    patient_id: str
    study_id: str
    file_reports: List[FileValidationReport] = field(default_factory=list)
    study_level_results: List[ValidationResult] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        """Check if any blocking issues in study or files."""
        if any(not r.passed and r.action == "block" for r in self.study_level_results):
            return True
        return any(f.has_blocking_issues for f in self.file_reports)


@dataclass
class PatientValidationReport:
    """Validation report for a patient (all studies)."""

    patient_id: str
    study_reports: List[StudyValidationReport] = field(default_factory=list)
    patient_level_results: List[ValidationResult] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        """Check if any blocking issues at patient or study level."""
        if any(not r.passed and r.action == "block" for r in self.patient_level_results):
            return True
        return any(s.has_blocking_issues for s in self.study_reports)


@dataclass
class QualityFilteringStats:
    """Statistics from quality filtering run."""

    total_files: int = 0
    files_passed: int = 0
    files_warned: int = 0
    files_blocked: int = 0
    studies_passed: int = 0
    studies_blocked: int = 0
    patients_passed: int = 0
    patients_blocked: int = 0
    issues_by_type: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# A: Data Validation Functions
# =============================================================================


def validate_nrrd_header(
    file_path: Path, config: QualityFilteringConfig
) -> ValidationResult:
    """A1: Validate NRRD file header.

    Checks:
    - File is readable
    - Contains valid 3D data (if require_3d is True)
    - Has space/orientation field (if require_space_field is True)

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "nrrd_validation"
    cfg = config.nrrd_validation

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        header = nrrd.read_header(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read NRRD header: {e}",
            action="block",
            details={"error": str(e)},
        )

    # Check dimension
    if cfg.require_3d:
        dimension = header.get("dimension", 0)
        if dimension != 3:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message=f"Expected 3D data, got {dimension}D",
                action="block",
                details={"dimension": dimension},
            )

    # Check space field
    if cfg.require_space_field:
        space = header.get("space", None)
        space_directions = header.get("space directions", None)
        if space is None and space_directions is None:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message="Missing space/orientation metadata in NRRD header",
                action="block",
                details={"space": space, "space_directions": space_directions},
            )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="NRRD header valid",
        details={"dimension": header.get("dimension"), "space": header.get("space")},
    )


def detect_scout_localizer(
    file_path: Path, config: QualityFilteringConfig
) -> ValidationResult:
    """A2: Detect scout/localizer images.

    Checks:
    - Minimum dimension in voxels
    - Maximum slice thickness

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "scout_detection"
    cfg = config.scout_detection

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        header = nrrd.read_header(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read header: {e}",
            action="block",
        )

    # Get dimensions
    sizes = header.get("sizes", [])
    if len(sizes) < 3:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Unexpected number of dimensions: {len(sizes)}",
            action="block",
            details={"sizes": sizes},
        )

    # Check minimum dimension
    min_dim = min(sizes[:3])
    if min_dim < cfg.min_dimension_voxels:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Dimension too small ({min_dim} < {cfg.min_dimension_voxels}), likely scout/localizer",
            action="block",
            details={"sizes": sizes, "min_dimension": min_dim},
        )

    # Check slice thickness
    space_directions = header.get("space directions", None)
    if space_directions is not None:
        try:
            spacings = [np.linalg.norm(d) for d in space_directions if d is not None]
            if spacings:
                max_spacing = max(spacings)
                if max_spacing > cfg.max_slice_thickness_mm:
                    return ValidationResult(
                        passed=False,
                        check_name=check_name,
                        message=f"Slice thickness too large ({max_spacing:.2f}mm > {cfg.max_slice_thickness_mm}mm)",
                        action="block",
                        details={"spacings": spacings, "max_spacing": max_spacing},
                    )
        except Exception:
            pass  # Skip spacing check if parsing fails

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Not a scout/localizer",
        details={"sizes": sizes},
    )


def check_voxel_spacing(
    file_path: Path, config: QualityFilteringConfig
) -> ValidationResult:
    """A3: Validate voxel spacing.

    Checks:
    - Spacing within min/max bounds
    - Anisotropy ratio within bounds

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "voxel_spacing"
    cfg = config.voxel_spacing

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        header = nrrd.read_header(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read header: {e}",
            action=cfg.action,
        )

    space_directions = header.get("space directions", None)
    if space_directions is None:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message="No spacing information available",
        )

    try:
        spacings = [np.linalg.norm(d) for d in space_directions if d is not None]
        if not spacings:
            return ValidationResult(
                passed=True, check_name=check_name, message="Could not parse spacings"
            )

        min_spacing = min(spacings)
        max_spacing = max(spacings)
        anisotropy = max_spacing / min_spacing if min_spacing > 0 else float("inf")

        issues = []

        if min_spacing < cfg.min_spacing_mm:
            issues.append(
                f"Min spacing {min_spacing:.3f}mm < {cfg.min_spacing_mm}mm"
            )

        if max_spacing > cfg.max_spacing_mm:
            issues.append(
                f"Max spacing {max_spacing:.3f}mm > {cfg.max_spacing_mm}mm"
            )

        if anisotropy > cfg.max_anisotropy_ratio:
            issues.append(
                f"Anisotropy {anisotropy:.2f} > {cfg.max_anisotropy_ratio}"
            )

        if issues:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message="; ".join(issues),
                action=cfg.action,
                details={
                    "spacings": spacings,
                    "min_spacing": min_spacing,
                    "max_spacing": max_spacing,
                    "anisotropy": anisotropy,
                },
            )

        return ValidationResult(
            passed=True,
            check_name=check_name,
            message="Voxel spacing within bounds",
            details={"spacings": spacings, "anisotropy": anisotropy},
        )

    except Exception as e:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message=f"Could not parse spacing: {e}",
        )


# =============================================================================
# B: Image Quality Functions
# =============================================================================


def compute_snr_background(data: np.ndarray) -> float:
    """Compute SNR using background noise estimation.

    Args:
        data: Image data array.

    Returns:
        Estimated SNR value.
    """
    # Use low-intensity voxels as background
    threshold = np.percentile(data[data > 0], 10) if np.any(data > 0) else 0
    background = data[data <= threshold]
    foreground = data[data > threshold]

    if len(background) == 0 or len(foreground) == 0:
        return 0.0

    noise_std = np.std(background)
    signal_mean = np.mean(foreground)

    if noise_std == 0:
        return float("inf")

    return signal_mean / noise_std


def check_snr(file_path: Path, config: QualityFilteringConfig) -> ValidationResult:
    """B1: Check signal-to-noise ratio.

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "snr_filtering"
    cfg = config.snr_filtering

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        data, _ = nrrd.read(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read file: {e}",
            action=cfg.action,
        )

    snr = compute_snr_background(data)

    if snr < cfg.min_snr:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"SNR {snr:.2f} < {cfg.min_snr}",
            action=cfg.action,
            details={"snr": snr},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message=f"SNR {snr:.2f} >= {cfg.min_snr}",
        details={"snr": snr},
    )


def check_contrast(file_path: Path, config: QualityFilteringConfig) -> ValidationResult:
    """B2: Check image contrast (detect uniform images).

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "contrast_detection"
    cfg = config.contrast_detection

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        data, _ = nrrd.read(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read file: {e}",
            action=cfg.action,
        )

    data_flat = data.flatten()

    # Check std/mean ratio
    mean_val = np.mean(data_flat)
    std_val = np.std(data_flat)

    if mean_val == 0:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message="Image is all zeros",
            action=cfg.action,
            details={"mean": 0, "std": 0},
        )

    std_ratio = std_val / abs(mean_val)
    if std_ratio < cfg.min_std_ratio:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Low contrast: std/mean = {std_ratio:.4f} < {cfg.min_std_ratio}",
            action=cfg.action,
            details={"std_ratio": std_ratio, "mean": mean_val, "std": std_val},
        )

    # Check uniform fraction (most common value)
    unique, counts = np.unique(data_flat, return_counts=True)
    max_count_fraction = counts.max() / len(data_flat)

    if max_count_fraction > cfg.max_uniform_fraction:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Uniform image: {max_count_fraction:.1%} of voxels have same value",
            action=cfg.action,
            details={"uniform_fraction": max_count_fraction},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Adequate contrast",
        details={"std_ratio": std_ratio, "uniform_fraction": max_count_fraction},
    )


def check_intensity_outliers(
    file_path: Path, config: QualityFilteringConfig
) -> ValidationResult:
    """B3: Check for intensity outliers (NaN, Inf, extreme values).

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "intensity_outliers"
    cfg = config.intensity_outliers

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        data, _ = nrrd.read(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read file: {e}",
            action=cfg.action,
        )

    # Check for NaN/Inf
    if cfg.reject_nan_inf:
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        if nan_count > 0 or inf_count > 0:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message=f"Found {nan_count} NaN and {inf_count} Inf values",
                action="block",  # Always block NaN/Inf
                details={"nan_count": int(nan_count), "inf_count": int(inf_count)},
            )

    # Check for extreme outliers
    data_valid = data[np.isfinite(data)]
    if len(data_valid) == 0:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message="No valid (finite) values in image",
            action="block",
        )

    max_val = np.max(data_valid)
    p99 = np.percentile(data_valid, 99)

    if p99 > 0 and max_val / p99 > cfg.max_outlier_ratio:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Extreme outlier: max/99th = {max_val/p99:.1f} > {cfg.max_outlier_ratio}",
            action=cfg.action,
            details={"max": float(max_val), "p99": float(p99)},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="No intensity outliers",
        details={"max": float(max_val), "p99": float(p99)},
    )


# =============================================================================
# C: Geometric Validation Functions
# =============================================================================


def validate_affine(file_path: Path, config: QualityFilteringConfig) -> ValidationResult:
    """C1: Validate affine transformation matrix.

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "affine_validation"
    cfg = config.affine_validation

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        header = nrrd.read_header(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read header: {e}",
            action=cfg.action,
        )

    space_directions = header.get("space directions", None)
    if space_directions is None:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message="No affine information available",
        )

    try:
        # Build rotation/scaling matrix from space directions
        matrix = np.array([d for d in space_directions if d is not None])
        if matrix.shape[0] < 3:
            return ValidationResult(
                passed=True,
                check_name=check_name,
                message="Incomplete space directions",
            )

        det = np.linalg.det(matrix)
        abs_det = abs(det)

        # Check for NaN/Inf in matrix
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message="Affine matrix contains NaN or Inf",
                action=cfg.action,
                details={"determinant": float(det)},
            )

        # Check determinant bounds
        if abs_det < cfg.min_det:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message=f"Affine determinant too small: |{det:.6f}| < {cfg.min_det}",
                action=cfg.action,
                details={"determinant": float(det)},
            )

        if abs_det > cfg.max_det:
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message=f"Affine determinant too large: |{det:.6f}| > {cfg.max_det}",
                action=cfg.action,
                details={"determinant": float(det)},
            )

        return ValidationResult(
            passed=True,
            check_name=check_name,
            message="Affine matrix valid",
            details={"determinant": float(det)},
        )

    except Exception as e:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message=f"Could not validate affine: {e}",
        )


def check_fov_consistency(
    file_path: Path, config: QualityFilteringConfig
) -> ValidationResult:
    """C2: Check field-of-view consistency (asymmetry).

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "fov_consistency"
    cfg = config.fov_consistency

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    try:
        header = nrrd.read_header(str(file_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Failed to read header: {e}",
            action="block",
        )

    sizes = header.get("sizes", [])
    space_directions = header.get("space directions", None)

    if len(sizes) < 3:
        return ValidationResult(
            passed=True, check_name=check_name, message="Insufficient size information"
        )

    # Calculate FOV in mm
    if space_directions is not None:
        try:
            spacings = [np.linalg.norm(d) for d in space_directions if d is not None]
            fov = [sizes[i] * spacings[i] for i in range(min(3, len(spacings)))]
        except Exception:
            fov = sizes[:3]  # Use voxel counts as fallback
    else:
        fov = sizes[:3]

    min_fov = min(fov)
    max_fov = max(fov)
    ratio = max_fov / min_fov if min_fov > 0 else float("inf")

    if ratio > cfg.block_ratio:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Extreme FOV asymmetry: ratio {ratio:.2f} > {cfg.block_ratio}",
            action="block",
            details={"fov": fov, "ratio": ratio},
        )

    if ratio > cfg.warn_ratio:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"High FOV asymmetry: ratio {ratio:.2f} > {cfg.warn_ratio}",
            action="warn",
            details={"fov": fov, "ratio": ratio},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="FOV within acceptable range",
        details={"fov": fov, "ratio": ratio},
    )


def check_orientation_consistency(
    study_files: Dict[str, Path], config: QualityFilteringConfig
) -> ValidationResult:
    """C3: Check orientation consistency within a study.

    Args:
        study_files: Dictionary mapping modality names to file paths.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "orientation_consistency"
    cfg = config.orientation_consistency

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    orientations = {}
    for modality, file_path in study_files.items():
        try:
            header = nrrd.read_header(str(file_path))
            space = header.get("space", "unknown")
            orientations[modality] = space
        except Exception:
            orientations[modality] = "error"

    unique_orientations = set(
        v for v in orientations.values() if v not in ("unknown", "error")
    )

    if len(unique_orientations) > 1:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Inconsistent orientations: {unique_orientations}",
            action=cfg.action,
            details={"orientations": orientations},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Orientations consistent",
        details={"orientations": orientations},
    )


# =============================================================================
# D: Longitudinal Validation Functions
# =============================================================================


def check_temporal_ordering(
    patient_id: str,
    study_ids: List[str],
    metadata_manager: Optional[Any] = None,
    config: Optional[QualityFilteringConfig] = None,
) -> ValidationResult:
    """D1: Check temporal ordering of studies.

    Args:
        patient_id: Patient identifier.
        study_ids: List of study IDs in directory order.
        metadata_manager: Optional metadata manager with acquisition dates.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "temporal_ordering"

    if config is None or not config.temporal_ordering.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    # Parse study numbers from IDs (e.g., MenGrowth-0001-002 -> 2)
    try:
        study_numbers = []
        for sid in study_ids:
            parts = sid.split("-")
            if len(parts) >= 3:
                study_numbers.append(int(parts[-1]))
            else:
                study_numbers.append(-1)

        # Check if sorted
        if study_numbers != sorted(study_numbers):
            return ValidationResult(
                passed=False,
                check_name=check_name,
                message=f"Study IDs not in temporal order: {study_ids}",
                action="warn",
                details={"study_ids": study_ids, "parsed_numbers": study_numbers},
            )

    except Exception as e:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message=f"Could not parse study IDs: {e}",
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Studies in temporal order",
        details={"study_ids": study_ids},
    )


def check_anatomical_consistency(
    patient_volumes: Dict[str, float], config: QualityFilteringConfig
) -> ValidationResult:
    """D2: Check anatomical consistency across timepoints.

    Note: This requires brain volumes which are computed during preprocessing.
    At curation stage, this check uses rough estimates from image dimensions.

    Args:
        patient_volumes: Dictionary mapping study_id to estimated brain volume (cc).
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "anatomical_consistency"
    cfg = config.anatomical_consistency

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    if len(patient_volumes) < 2:
        return ValidationResult(
            passed=True,
            check_name=check_name,
            message="Need at least 2 studies for comparison",
        )

    volumes = list(patient_volumes.values())
    issues = []

    # Check volume bounds
    for study_id, vol in patient_volumes.items():
        if vol < cfg.min_brain_volume_cc:
            issues.append(f"{study_id}: volume {vol:.0f}cc < {cfg.min_brain_volume_cc}cc")
        if vol > cfg.max_brain_volume_cc:
            issues.append(f"{study_id}: volume {vol:.0f}cc > {cfg.max_brain_volume_cc}cc")

    # Check volume changes between consecutive studies
    sorted_studies = sorted(patient_volumes.keys())
    for i in range(1, len(sorted_studies)):
        prev_vol = patient_volumes[sorted_studies[i - 1]]
        curr_vol = patient_volumes[sorted_studies[i]]
        if prev_vol > 0:
            change_pct = abs(curr_vol - prev_vol) / prev_vol * 100
            if change_pct > cfg.max_volume_change_percent:
                issues.append(
                    f"{sorted_studies[i-1]}->{sorted_studies[i]}: {change_pct:.1f}% change"
                )

    if issues:
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message="; ".join(issues),
            action=cfg.action,
            details={"volumes": patient_volumes},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Anatomical consistency OK",
        details={"volumes": patient_volumes},
    )


def check_modality_consistency(
    patient_modalities: Dict[str, List[str]], config: QualityFilteringConfig
) -> ValidationResult:
    """D3: Check modality consistency across timepoints.

    Args:
        patient_modalities: Dict mapping study_id to list of modalities.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "modality_consistency"
    cfg = config.modality_consistency

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    modality_sets = [set(m) for m in patient_modalities.values()]
    if not modality_sets:
        return ValidationResult(
            passed=True, check_name=check_name, message="No modalities to compare"
        )

    # Check if all studies have same modalities
    first_set = modality_sets[0]
    if not all(s == first_set for s in modality_sets):
        return ValidationResult(
            passed=False,
            check_name=check_name,
            message=f"Inconsistent modalities across timepoints",
            action=cfg.action,
            details={"modalities_per_study": patient_modalities},
        )

    return ValidationResult(
        passed=True,
        check_name=check_name,
        message="Modalities consistent across timepoints",
        details={"modalities_per_study": patient_modalities},
    )


# =============================================================================
# E: Preprocessing Checks
# =============================================================================


def check_registration_reference(
    study_modalities: List[str], config: QualityFilteringConfig
) -> ValidationResult:
    """E1: Check if a valid registration reference modality exists.

    Args:
        study_modalities: List of modalities in the study.
        config: Quality filtering configuration.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    check_name = "registration_reference"
    cfg = config.registration_reference

    if not cfg.enabled:
        return ValidationResult(
            passed=True, check_name=check_name, message="Check disabled"
        )

    # Parse priority string (e.g., "t1n > t1c > t2f > t2w")
    priority_list = [m.strip() for m in cfg.priority.split(">")]

    # Find first available modality from priority list
    for modality in priority_list:
        if modality in study_modalities:
            return ValidationResult(
                passed=True,
                check_name=check_name,
                message=f"Reference modality available: {modality}",
                details={"reference": modality, "available": study_modalities},
            )

    return ValidationResult(
        passed=False,
        check_name=check_name,
        message=f"No reference modality found. Need one of: {priority_list}",
        action=cfg.action,
        details={"priority": priority_list, "available": study_modalities},
    )


# =============================================================================
# Main Quality Filtering Function
# =============================================================================


def validate_file(
    file_path: Path,
    config: QualityFilteringConfig,
    load_data: bool = True,
) -> FileValidationReport:
    """Run all file-level validation checks on a single file.

    Args:
        file_path: Path to NRRD file.
        config: Quality filtering configuration.
        load_data: Whether to load image data (for quality checks).

    Returns:
        FileValidationReport with all check results.
    """
    # Parse file info
    parts = file_path.parent.name.split("-")
    if len(parts) >= 3:
        patient_id = "-".join(parts[:2])
        study_id = file_path.parent.name
    else:
        patient_id = file_path.parent.parent.name
        study_id = file_path.parent.name

    modality = file_path.stem

    report = FileValidationReport(
        file_path=file_path,
        patient_id=patient_id,
        study_id=study_id,
        modality=modality,
    )

    # A: Data Validation
    report.results.append(validate_nrrd_header(file_path, config))
    report.results.append(detect_scout_localizer(file_path, config))
    report.results.append(check_voxel_spacing(file_path, config))

    # B: Image Quality (requires loading data)
    if load_data:
        report.results.append(check_snr(file_path, config))
        report.results.append(check_contrast(file_path, config))
        report.results.append(check_intensity_outliers(file_path, config))

    # C: Geometric
    report.results.append(validate_affine(file_path, config))
    report.results.append(check_fov_consistency(file_path, config))

    return report


def run_quality_filtering(
    data_root: Path,
    config: QualityFilteringConfig,
    metadata_manager: Optional[Any] = None,
    dry_run: bool = False,
) -> Tuple[QualityFilteringStats, List[PatientValidationReport]]:
    """Run complete quality filtering on dataset.

    Args:
        data_root: Path to MenGrowth-2025 directory.
        config: Quality filtering configuration.
        metadata_manager: Optional metadata manager for additional context.
        dry_run: If True, don't delete files.

    Returns:
        Tuple of (stats, patient_reports).
    """
    if not config.enabled:
        logger.info("Quality filtering is disabled")
        return QualityFilteringStats(), []

    stats = QualityFilteringStats()
    patient_reports: List[PatientValidationReport] = []

    # Find all patient directories
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(patient_dirs)} patients to validate")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        patient_report = PatientValidationReport(patient_id=patient_id)

        # Find all study directories
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        patient_modalities: Dict[str, List[str]] = {}

        for study_dir in study_dirs:
            study_id = study_dir.name
            study_report = StudyValidationReport(
                patient_id=patient_id, study_id=study_id
            )

            # Find all NRRD files
            nrrd_files = list(study_dir.glob("*.nrrd"))
            study_files = {f.stem: f for f in nrrd_files}
            patient_modalities[study_id] = list(study_files.keys())

            for file_path in nrrd_files:
                stats.total_files += 1
                file_report = validate_file(file_path, config)
                study_report.file_reports.append(file_report)

                if file_report.has_blocking_issues:
                    stats.files_blocked += 1
                    for issue in file_report.blocking_issues:
                        issue_type = issue.check_name
                        stats.issues_by_type[issue_type] = (
                            stats.issues_by_type.get(issue_type, 0) + 1
                        )
                elif file_report.has_warnings:
                    stats.files_warned += 1
                else:
                    stats.files_passed += 1

            # Study-level checks
            if study_files:
                # C3: Orientation consistency
                orient_result = check_orientation_consistency(study_files, config)
                study_report.study_level_results.append(orient_result)

                # E1: Registration reference
                ref_result = check_registration_reference(
                    list(study_files.keys()), config
                )
                study_report.study_level_results.append(ref_result)

            if study_report.has_blocking_issues:
                stats.studies_blocked += 1
            else:
                stats.studies_passed += 1

            patient_report.study_reports.append(study_report)

        # Patient-level checks
        if len(study_dirs) >= 2:
            # D1: Temporal ordering
            study_ids = [d.name for d in study_dirs]
            temporal_result = check_temporal_ordering(
                patient_id, study_ids, metadata_manager, config
            )
            patient_report.patient_level_results.append(temporal_result)

            # D3: Modality consistency
            modality_result = check_modality_consistency(patient_modalities, config)
            patient_report.patient_level_results.append(modality_result)

        if patient_report.has_blocking_issues:
            stats.patients_blocked += 1
        else:
            stats.patients_passed += 1

        patient_reports.append(patient_report)

    logger.info(f"Quality filtering complete:")
    logger.info(f"  Files: {stats.files_passed} passed, {stats.files_warned} warned, {stats.files_blocked} blocked")
    logger.info(f"  Studies: {stats.studies_passed} passed, {stats.studies_blocked} blocked")
    logger.info(f"  Patients: {stats.patients_passed} passed, {stats.patients_blocked} blocked")

    return stats, patient_reports


def export_quality_issues(
    patient_reports: List[PatientValidationReport],
    output_path: Path,
) -> None:
    """Export quality issues to CSV.

    Args:
        patient_reports: List of patient validation reports.
        output_path: Path to output CSV file.
    """
    import csv

    rows = []
    for patient_report in patient_reports:
        for study_report in patient_report.study_reports:
            # File-level issues
            for file_report in study_report.file_reports:
                for result in file_report.results:
                    if not result.passed:
                        rows.append({
                            "patient_id": patient_report.patient_id,
                            "study_id": study_report.study_id,
                            "modality": file_report.modality,
                            "file_path": str(file_report.file_path),
                            "check_name": result.check_name,
                            "action": result.action,
                            "message": result.message,
                            "level": "file",
                        })

            # Study-level issues
            for result in study_report.study_level_results:
                if not result.passed:
                    rows.append({
                        "patient_id": patient_report.patient_id,
                        "study_id": study_report.study_id,
                        "modality": "",
                        "file_path": "",
                        "check_name": result.check_name,
                        "action": result.action,
                        "message": result.message,
                        "level": "study",
                    })

        # Patient-level issues
        for result in patient_report.patient_level_results:
            if not result.passed:
                rows.append({
                    "patient_id": patient_report.patient_id,
                    "study_id": "",
                    "modality": "",
                    "file_path": "",
                    "check_name": result.check_name,
                    "action": result.action,
                    "message": result.message,
                    "level": "patient",
                })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        else:
            # Write empty file with headers
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "patient_id",
                    "study_id",
                    "modality",
                    "file_path",
                    "check_name",
                    "action",
                    "message",
                    "level",
                ],
            )
            writer.writeheader()

    logger.info(f"Exported {len(rows)} quality issues to {output_path}")
