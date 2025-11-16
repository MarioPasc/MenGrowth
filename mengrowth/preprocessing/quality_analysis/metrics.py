"""Metric computation functions for MRI dataset quality analysis.

This module provides functions for computing various quality metrics on MRI datasets,
including spatial properties, intensity statistics, and consistency checks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def load_image(
    image_path: Path, file_format: str = "auto"
) -> Optional[sitk.Image]:
    """Load medical image from NRRD or NIfTI format using SimpleITK.

    Args:
        image_path: Path to image file (.nrrd or .nii.gz).
        file_format: Format hint ('auto', 'nrrd', 'nifti').

    Returns:
        SimpleITK Image object, or None if loading fails.

    Raises:
        RuntimeError: If image loading fails (handled by caller).

    Examples:
        >>> img = load_image(Path("scan.nrrd"))
        >>> print(img.GetSize())
        (256, 256, 180)
    """
    try:
        image = sitk.ReadImage(str(image_path))
        logger.debug(f"Successfully loaded image: {image_path.name}")
        return image
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def compute_voxel_spacing(image: sitk.Image) -> Tuple[float, float, float]:
    """Extract physical voxel spacing from image metadata.

    Args:
        image: SimpleITK Image object.

    Returns:
        Tuple of (x_spacing, y_spacing, z_spacing) in millimeters.

    Examples:
        >>> img = load_image(Path("scan.nrrd"))
        >>> spacing = compute_voxel_spacing(img)
        >>> print(f"Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm")
        Spacing: 0.94 x 0.94 x 1.00 mm
    """
    spacing = image.GetSpacing()
    return tuple(spacing)


def compute_image_dimensions(image: sitk.Image) -> Tuple[int, int, int]:
    """Extract image dimensions (shape).

    Args:
        image: SimpleITK Image object.

    Returns:
        Tuple of (width, height, depth) in voxels.

    Examples:
        >>> img = load_image(Path("scan.nrrd"))
        >>> dims = compute_image_dimensions(img)
        >>> print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
        Dimensions: 256 x 256 x 180
    """
    size = image.GetSize()
    return tuple(size)


def compute_intensity_statistics(
    image: sitk.Image, percentiles: List[int]
) -> Dict[str, float]:
    """Compute comprehensive intensity value statistics.

    Args:
        image: SimpleITK Image object.
        percentiles: List of percentiles to compute (e.g., [1, 5, 25, 50, 75, 95, 99]).

    Returns:
        Dictionary containing:
            - min, max, mean, std: Basic statistics
            - p1, p5, ...: Requested percentiles
            - range: max - min
            - non_zero_mean: Mean of non-zero values
            - non_zero_fraction: Fraction of non-zero voxels

    Examples:
        >>> img = load_image(Path("t1c.nrrd"))
        >>> stats = compute_intensity_statistics(img, [1, 50, 99])
        >>> print(f"Mean intensity: {stats['mean']:.2f}")
        Mean intensity: 245.32
    """
    # Convert to numpy array for efficient computation
    array = sitk.GetArrayFromImage(image)

    stats = {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "range": float(np.max(array) - np.min(array)),
    }

    # Compute percentiles
    percentile_values = np.percentile(array, percentiles)
    for p, val in zip(percentiles, percentile_values):
        stats[f"p{p}"] = float(val)

    # Non-zero statistics (useful for MRI where background is often zero)
    non_zero_array = array[array > 0]
    if len(non_zero_array) > 0:
        stats["non_zero_mean"] = float(np.mean(non_zero_array))
        stats["non_zero_std"] = float(np.std(non_zero_array))
        stats["non_zero_fraction"] = float(len(non_zero_array) / array.size)
    else:
        stats["non_zero_mean"] = 0.0
        stats["non_zero_std"] = 0.0
        stats["non_zero_fraction"] = 0.0

    return stats


def detect_outliers_iqr(
    values: np.ndarray, multiplier: float = 1.5
) -> Tuple[List[int], float, float]:
    """Detect outliers using Interquartile Range (IQR) method.

    Args:
        values: Array of numerical values.
        multiplier: IQR multiplier (typically 1.5 for outliers, 3.0 for extreme outliers).

    Returns:
        Tuple of (outlier_indices, lower_bound, upper_bound).

    Examples:
        >>> values = np.array([1, 2, 3, 4, 5, 100])
        >>> outliers, lower, upper = detect_outliers_iqr(values)
        >>> print(f"Found {len(outliers)} outliers")
        Found 1 outliers
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
    return outlier_indices.tolist(), float(lower_bound), float(upper_bound)


def detect_outliers_zscore(
    values: np.ndarray, threshold: float = 3.0
) -> Tuple[List[int], float, float]:
    """Detect outliers using Z-score method.

    Args:
        values: Array of numerical values.
        threshold: Z-score threshold (typically 3.0).

    Returns:
        Tuple of (outlier_indices, lower_bound, upper_bound).

    Examples:
        >>> values = np.array([1, 2, 3, 4, 5, 100])
        >>> outliers, lower, upper = detect_outliers_zscore(values)
        >>> print(f"Found {len(outliers)} outliers")
        Found 1 outliers
    """
    mean = np.mean(values)
    std = np.std(values)

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
    return outlier_indices.tolist(), float(lower_bound), float(upper_bound)


def compute_snr_estimate(image: sitk.Image) -> float:
    """Estimate signal-to-noise ratio using background-based method.

    This is a simplified SNR estimation that assumes:
    - Background voxels have values close to zero
    - Signal is the mean of non-zero voxels
    - Noise is the std of low-intensity voxels (bottom 5th percentile)

    Args:
        image: SimpleITK Image object.

    Returns:
        Estimated SNR value (signal_mean / noise_std).

    Examples:
        >>> img = load_image(Path("t1c.nrrd"))
        >>> snr = compute_snr_estimate(img)
        >>> print(f"Estimated SNR: {snr:.2f}")
        Estimated SNR: 23.45
    """
    array = sitk.GetArrayFromImage(image)

    # Signal: mean of non-zero voxels
    non_zero_array = array[array > 0]
    if len(non_zero_array) == 0:
        return 0.0

    signal = np.mean(non_zero_array)

    # Noise: estimate from low-intensity voxels (background)
    # Use bottom 5th percentile as background
    threshold = np.percentile(array, 5)
    background = array[array <= threshold]

    if len(background) == 0 or np.std(background) == 0:
        # Fallback: use std of all voxels
        noise_std = np.std(array)
    else:
        noise_std = np.std(background)

    if noise_std == 0:
        return float('inf')

    snr = signal / noise_std
    return float(snr)


def compute_contrast_ratio(
    image: sitk.Image, roi1_percentile: int = 75, roi2_percentile: int = 25
) -> float:
    """Compute contrast ratio between high and low intensity regions.

    Args:
        image: SimpleITK Image object.
        roi1_percentile: Percentile for high-intensity region (default 75).
        roi2_percentile: Percentile for low-intensity region (default 25).

    Returns:
        Contrast ratio: (mean_roi1 - mean_roi2) / (mean_roi1 + mean_roi2).

    Examples:
        >>> img = load_image(Path("t1c.nrrd"))
        >>> contrast = compute_contrast_ratio(img)
        >>> print(f"Contrast ratio: {contrast:.3f}")
        Contrast ratio: 0.456
    """
    array = sitk.GetArrayFromImage(image)

    # Define ROIs based on percentiles
    threshold_high = np.percentile(array, roi1_percentile)
    threshold_low = np.percentile(array, roi2_percentile)

    roi1 = array[array >= threshold_high]
    roi2 = array[array <= threshold_low]

    if len(roi1) == 0 or len(roi2) == 0:
        return 0.0

    mean_roi1 = np.mean(roi1)
    mean_roi2 = np.mean(roi2)

    denominator = mean_roi1 + mean_roi2
    if denominator == 0:
        return 0.0

    contrast = (mean_roi1 - mean_roi2) / denominator
    return float(contrast)


def compute_histogram(
    image: sitk.Image, bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute intensity histogram.

    Args:
        image: SimpleITK Image object.
        bins: Number of histogram bins.

    Returns:
        Tuple of (histogram_counts, bin_edges).

    Examples:
        >>> img = load_image(Path("t1c.nrrd"))
        >>> counts, edges = compute_histogram(img, bins=50)
        >>> print(f"Histogram has {len(counts)} bins")
        Histogram has 50 bins
    """
    array = sitk.GetArrayFromImage(image)
    counts, edges = np.histogram(array.flatten(), bins=bins)
    return counts, edges


def compute_patient_statistics(
    patient_studies: Dict[str, List[str]]
) -> Dict[str, any]:
    """Compute statistics about studies per patient.

    Args:
        patient_studies: Dictionary mapping patient_id -> list of study names.

    Returns:
        Dictionary containing:
            - total_patients: Number of unique patients
            - total_studies: Total number of studies
            - studies_per_patient: List of study counts per patient
            - mean_studies: Mean number of studies per patient
            - std_studies: Standard deviation of studies per patient
            - min_studies: Minimum studies per patient
            - max_studies: Maximum studies per patient

    Examples:
        >>> patient_studies = {
        ...     'MenGrowth-0001': ['000', '001', '002'],
        ...     'MenGrowth-0002': ['000', '001']
        ... }
        >>> stats = compute_patient_statistics(patient_studies)
        >>> print(f"Mean studies per patient: {stats['mean_studies']:.2f}")
        Mean studies per patient: 2.50
    """
    studies_per_patient = [len(studies) for studies in patient_studies.values()]

    stats = {
        "total_patients": len(patient_studies),
        "total_studies": sum(studies_per_patient),
        "studies_per_patient": studies_per_patient,
        "mean_studies": float(np.mean(studies_per_patient)),
        "std_studies": float(np.std(studies_per_patient)),
        "min_studies": int(np.min(studies_per_patient)),
        "max_studies": int(np.max(studies_per_patient)),
    }

    return stats


def compute_missing_sequences(
    study_sequences: Dict[str, List[str]], expected_sequences: List[str]
) -> Dict[str, any]:
    """Compute statistics about missing sequences across studies.

    Args:
        study_sequences: Dictionary mapping study_id -> list of available sequences.
        expected_sequences: List of expected sequence names.

    Returns:
        Dictionary containing per-sequence statistics:
            - For each sequence: count, missing_count, missing_fraction
            - overall_missing_rate: Average fraction of missing sequences

    Examples:
        >>> study_sequences = {
        ...     'MenGrowth-0001-000': ['t1c', 't1n', 't2w'],
        ...     'MenGrowth-0001-001': ['t1c', 't2w']  # missing t1n
        ... }
        >>> stats = compute_missing_sequences(study_sequences, ['t1c', 't1n', 't2w'])
        >>> print(f"t1n missing in {stats['t1n']['missing_count']} studies")
        t1n missing in 1 studies
    """
    total_studies = len(study_sequences)
    stats = {}

    for sequence in expected_sequences:
        present_count = sum(
            1 for sequences in study_sequences.values() if sequence in sequences
        )
        missing_count = total_studies - present_count

        stats[sequence] = {
            "present_count": present_count,
            "missing_count": missing_count,
            "missing_fraction": float(missing_count / total_studies) if total_studies > 0 else 0.0,
        }

    # Overall missing rate
    total_expected = len(expected_sequences) * total_studies
    total_present = sum(stat["present_count"] for stat in stats.values())
    total_missing = total_expected - total_present

    stats["overall"] = {
        "total_expected": total_expected,
        "total_present": total_present,
        "total_missing": total_missing,
        "overall_missing_rate": float(total_missing / total_expected) if total_expected > 0 else 0.0,
    }

    return stats


def compute_spacing_statistics(
    spacings: List[Tuple[float, float, float]]
) -> Dict[str, any]:
    """Compute statistics for voxel spacing distributions.

    Args:
        spacings: List of (x, y, z) spacing tuples in millimeters.

    Returns:
        Dictionary with statistics for each axis (x, y, z):
            - mean, std, min, max, median

    Examples:
        >>> spacings = [(0.94, 0.94, 1.0), (1.0, 1.0, 1.0), (0.94, 0.94, 1.5)]
        >>> stats = compute_spacing_statistics(spacings)
        >>> print(f"Mean X spacing: {stats['x']['mean']:.3f} mm")
        Mean X spacing: 0.960 mm
    """
    if not spacings:
        return {}

    spacings_array = np.array(spacings)  # Shape: (N, 3)

    stats = {}
    for idx, axis in enumerate(['x', 'y', 'z']):
        axis_values = spacings_array[:, idx]
        stats[axis] = {
            "mean": float(np.mean(axis_values)),
            "std": float(np.std(axis_values)),
            "min": float(np.min(axis_values)),
            "max": float(np.max(axis_values)),
            "median": float(np.median(axis_values)),
        }

    return stats


def compute_dimension_statistics(
    dimensions: List[Tuple[int, int, int]]
) -> Dict[str, any]:
    """Compute statistics for image dimension distributions.

    Args:
        dimensions: List of (width, height, depth) tuples in voxels.

    Returns:
        Dictionary with statistics for each dimension:
            - mean, std, min, max, median

    Examples:
        >>> dimensions = [(256, 256, 180), (512, 512, 200), (256, 256, 180)]
        >>> stats = compute_dimension_statistics(dimensions)
        >>> print(f"Mean width: {stats['width']['mean']:.1f} voxels")
        Mean width: 341.3 voxels
    """
    if not dimensions:
        return {}

    dimensions_array = np.array(dimensions)  # Shape: (N, 3)

    stats = {}
    for idx, dim_name in enumerate(['width', 'height', 'depth']):
        dim_values = dimensions_array[:, idx]
        stats[dim_name] = {
            "mean": float(np.mean(dim_values)),
            "std": float(np.std(dim_values)),
            "min": int(np.min(dim_values)),
            "max": int(np.max(dim_values)),
            "median": float(np.median(dim_values)),
        }

    return stats
