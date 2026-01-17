"""Metric computation functions for per-step QC.

This module provides functions for:
1. Geometry/header consistency checks
2. Registration similarity metrics (NMI, NCC)
3. Mask plausibility metrics
4. Intensity stability (median/IQR, Wasserstein)
5. Downsampling and mask resolution
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import SimpleITK as sitk
from scipy.stats import wasserstein_distance
from scipy.ndimage import sobel, binary_erosion


# =====================================================================
# Downsampling and Masking Utilities
# =====================================================================

def downsample_image_for_qc(
    image: sitk.Image,
    target_mm: float = 2.0,
    max_voxels: int = 250000,
    seed: int = 1234
) -> Tuple[sitk.Image, float]:
    """Downsample image for cheap QC computation.

    Args:
        image: Input SimpleITK image
        target_mm: Target isotropic resolution in mm
        max_voxels: Maximum number of voxels (fallback to further downsampling)
        seed: Random seed for reproducibility

    Returns:
        (downsampled_image, downsample_factor)
    """
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    # Compute target spacing
    target_spacing = np.array([target_mm, target_mm, target_mm])

    # Compute new size
    new_size = (original_spacing / target_spacing * original_size).astype(int)

    # Check if exceeds max_voxels
    total_voxels = np.prod(new_size)
    if total_voxels > max_voxels:
        # Further downsample to meet max_voxels constraint
        scale = (max_voxels / total_voxels) ** (1/3)
        new_size = (new_size * scale).astype(int)
        target_spacing = (original_spacing * original_size / new_size)

    # Resample using linear interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputSpacing(target_spacing.tolist())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    downsampled = resampler.Execute(image)
    downsample_factor = float(target_spacing[0] / original_spacing[0])

    return downsampled, downsample_factor


def get_mask_for_qc(
    image: sitk.Image,
    mask_source: str,
    artifact_paths: Dict[str, Path],
    downsample_factor: float
) -> Optional[sitk.Image]:
    """Get or compute mask for QC metrics.

    Args:
        image: Downsampled image
        mask_source: "skullstrip_else_otsu" | "skullstrip_only" | "otsu_only" | "none"
        artifact_paths: Dict with potential "mask" path
        downsample_factor: Factor to downsample mask

    Returns:
        Binary mask as SimpleITK image, or None
    """
    if mask_source == "none":
        return None

    # Try to load skull-stripped mask
    if mask_source in ["skullstrip_else_otsu", "skullstrip_only"]:
        mask_path = artifact_paths.get("mask")
        if mask_path and mask_path.exists():
            try:
                mask = sitk.ReadImage(str(mask_path))
                # Downsample mask to match image
                return _downsample_mask(mask, image)
            except Exception:
                pass  # Fall through to Otsu if loading fails

    # Fallback to Otsu if allowed
    if mask_source in ["skullstrip_else_otsu", "otsu_only"]:
        return _compute_otsu_mask(image)

    return None


def _downsample_mask(mask: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    """Downsample binary mask to match reference image geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(mask)


def _compute_otsu_mask(image: sitk.Image) -> sitk.Image:
    """Compute Otsu threshold mask."""
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    return otsu_filter.Execute(image)


# =====================================================================
# Geometry Metrics
# =====================================================================

def compute_geometry_metrics(
    image: sitk.Image,
    config: Any
) -> Dict[str, Any]:
    """Compute geometry/header consistency metrics.

    Args:
        image: Input image
        config: QCGeometryMetricsConfig

    Returns:
        Dict with orientation, spacing, affine determinant
    """
    metrics = {}

    if config.check_orientation:
        # Extract orientation from direction matrix
        direction = np.array(image.GetDirection()).reshape(3, 3)
        # Simplified: check if close to identity or reflection
        metrics["orientation_det"] = float(np.linalg.det(direction))
        metrics["orientation_valid"] = abs(abs(metrics["orientation_det"]) - 1.0) < 0.01

    if config.check_spacing:
        spacing = image.GetSpacing()
        metrics["spacing_x"] = spacing[0]
        metrics["spacing_y"] = spacing[1]
        metrics["spacing_z"] = spacing[2]
        metrics["spacing_max_diff"] = max(spacing) - min(spacing)

    if config.check_affine_det:
        # Affine determinant checks for reflection/shearing
        direction = np.array(image.GetDirection()).reshape(3, 3)
        spacing_diag = np.diag(image.GetSpacing())
        affine_upper = direction @ spacing_diag
        metrics["affine_det"] = float(np.linalg.det(affine_upper))

    return metrics


# =====================================================================
# Registration Similarity Metrics
# =====================================================================

def compute_registration_similarity(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    mask: Optional[sitk.Image],
    config: Any,
    case_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute registration similarity metrics (NMI or NCC).

    Args:
        fixed_image: Reference image
        moving_image: Registered image
        mask: Optional binary mask to restrict computation
        config: QCRegistrationSimilarityConfig
        case_metadata: Dict with patient_id, modality, is_longitudinal, etc.

    Returns:
        Dict with NMI and/or NCC values
    """
    metrics = {}

    # Resample moving to fixed geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    moving_resampled = resampler.Execute(moving_image)

    # Convert to arrays
    fixed_arr = sitk.GetArrayFromImage(fixed_image)
    moving_arr = sitk.GetArrayFromImage(moving_resampled)

    # Apply mask if provided
    if config.use_mask and mask is not None:
        mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
        fixed_arr = fixed_arr[mask_arr]
        moving_arr = moving_arr[mask_arr]
    else:
        fixed_arr = fixed_arr.flatten()
        moving_arr = moving_arr.flatten()

    # Remove zero/invalid voxels
    valid = (fixed_arr > 0) & (moving_arr > 0)
    fixed_arr = fixed_arr[valid]
    moving_arr = moving_arr[valid]

    if len(fixed_arr) < 100:
        metrics["nmi"] = None
        metrics["ncc"] = None
        return metrics

    # Determine if same modality (for longitudinal) or multi-modal
    is_longitudinal = case_metadata.get("is_longitudinal", False)

    if config.nmi_multimodal and not is_longitudinal:
        # Normalized Mutual Information
        nmi = _compute_nmi(fixed_arr, moving_arr)
        metrics["nmi"] = nmi

    if config.ncc_longitudinal and is_longitudinal:
        # Normalized Cross-Correlation
        ncc = _compute_ncc(fixed_arr, moving_arr)
        metrics["ncc"] = ncc

    return metrics


def _compute_nmi(fixed: np.ndarray, moving: np.ndarray, bins: int = 64) -> float:
    """Compute Normalized Mutual Information."""
    hist_2d, _, _ = np.histogram2d(fixed, moving, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Entropy
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    # Marginal entropies
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

    nmi = 2 * mi / (hx + hy) if (hx + hy) > 0 else 0.0
    return float(nmi)


def _compute_ncc(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Compute Normalized Cross-Correlation."""
    fixed_norm = (fixed - np.mean(fixed)) / (np.std(fixed) + 1e-8)
    moving_norm = (moving - np.mean(moving)) / (np.std(moving) + 1e-8)
    ncc = np.mean(fixed_norm * moving_norm)
    return float(ncc)


# =====================================================================
# Mask Plausibility Metrics
# =====================================================================

def compute_mask_plausibility(
    mask: sitk.Image,
    image: sitk.Image,
    config: Any
) -> Dict[str, Any]:
    """Compute mask plausibility metrics.

    Args:
        mask: Binary mask
        image: Corresponding image
        config: QCMaskPlausibilityConfig

    Returns:
        Dict with volume, boundary gradient score
    """
    metrics = {}
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    image_arr = sitk.GetArrayFromImage(image)

    if config.check_volume:
        voxel_volume = np.prod(mask.GetSpacing())  # mm^3 per voxel
        brain_volume_cc = np.sum(mask_arr) * voxel_volume / 1000  # convert to cc
        metrics["mask_volume_cc"] = float(brain_volume_cc)

    if config.boundary_gradient_score:
        # Compute gradient magnitude at mask boundary
        gradient_mag = np.sqrt(
            sobel(image_arr, axis=0)**2 +
            sobel(image_arr, axis=1)**2 +
            sobel(image_arr, axis=2)**2
        )

        # Erode mask to get boundary
        eroded = binary_erosion(mask_arr, iterations=1)
        boundary = mask_arr & ~eroded

        if np.sum(boundary) > 0 and np.sum(mask_arr) > 0:
            mean_boundary_gradient = np.mean(gradient_mag[boundary])
            mean_inside_gradient = np.mean(gradient_mag[mask_arr & ~boundary])
            if mean_inside_gradient > 0:
                metrics["boundary_gradient_score"] = float(mean_boundary_gradient / mean_inside_gradient)
            else:
                metrics["boundary_gradient_score"] = 0.0
        else:
            metrics["boundary_gradient_score"] = 0.0

    return metrics


def compute_longitudinal_mask_dice(
    ref_mask_path: Path,
    warped_mask_path: Path
) -> float:
    """Compute Dice coefficient between reference and warped mask.

    Args:
        ref_mask_path: Path to reference mask
        warped_mask_path: Path to warped mask

    Returns:
        Dice coefficient (0-1)
    """
    ref_mask = sitk.ReadImage(str(ref_mask_path))
    warped_mask = sitk.ReadImage(str(warped_mask_path))

    ref_arr = sitk.GetArrayFromImage(ref_mask).astype(bool)
    warped_arr = sitk.GetArrayFromImage(warped_mask).astype(bool)

    intersection = np.sum(ref_arr & warped_arr)
    union = np.sum(ref_arr) + np.sum(warped_arr)

    if union == 0:
        return 0.0

    dice = 2 * intersection / union
    return float(dice)


# =====================================================================
# Intensity Stability Metrics
# =====================================================================

def compute_intensity_stats_for_wasserstein(
    image: sitk.Image,
    mask: Optional[sitk.Image],
    config: Any
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Compute intensity statistics and histogram for Wasserstein.

    Args:
        image: Input image
        mask: Optional mask
        config: QCIntensityStabilityConfig

    Returns:
        (metrics_dict, histogram_array, bin_edges)
    """
    metrics = {}
    image_arr = sitk.GetArrayFromImage(image)

    # Apply mask
    if mask is not None:
        mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
        values = image_arr[mask_arr]
    else:
        values = image_arr[image_arr > 0]

    if len(values) == 0:
        return {"median": None, "iqr": None}, np.array([]), np.array([])

    # Median and IQR
    if config.median_iqr:
        metrics["median"] = float(np.median(values))
        metrics["iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))

    # Compute histogram for Wasserstein
    hist = np.array([])
    bin_edges = np.array([])

    if config.wasserstein_distance:
        # Use percentile range to avoid extreme outliers
        p_low, p_high = config.histogram_range_percentiles
        vmin, vmax = np.percentile(values, [p_low, p_high])

        if vmax > vmin:
            hist, bin_edges = np.histogram(
                values,
                bins=config.histogram_bins,
                range=(vmin, vmax),
                density=True
            )

    return metrics, hist, bin_edges


def compute_reference_histogram(
    histograms: list,
    bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reference histogram from accumulated samples.

    Args:
        histograms: List of histogram arrays (same bins)
        bin_edges: Bin edges (same for all histograms)

    Returns:
        (mean_histogram, bin_edges)
    """
    if len(histograms) == 0:
        return np.array([]), np.array([])

    # Average histograms
    hist_stack = np.stack(histograms, axis=0)
    mean_hist = np.mean(hist_stack, axis=0)

    return mean_hist, bin_edges


def compute_wasserstein_distance(
    case_hist: np.ndarray,
    ref_hist: np.ndarray,
    bin_edges: np.ndarray
) -> float:
    """Compute Wasserstein distance between two histograms.

    Args:
        case_hist: Case histogram (density)
        ref_hist: Reference histogram (density)
        bin_edges: Bin edges for both histograms

    Returns:
        Wasserstein distance
    """
    if len(case_hist) == 0 or len(ref_hist) == 0 or len(bin_edges) < 2:
        return 0.0

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Use scipy's wasserstein_distance with weighted distributions
    wass_dist = wasserstein_distance(bin_centers, bin_centers, case_hist, ref_hist)
    return float(wass_dist)


# =====================================================================
# Outlier Detection
# =====================================================================

def detect_outliers_mad(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Detect outliers using Median Absolute Deviation (robust).

    Args:
        values: Array of values
        threshold: MAD threshold (typically 3.5)

    Returns:
        Boolean array indicating outliers
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return np.zeros(len(values), dtype=bool)

    modified_z_scores = 0.6745 * (values - median) / mad
    return np.abs(modified_z_scores) > threshold


def detect_outliers_iqr(values: np.ndarray, multiplier: float = 3.0) -> np.ndarray:
    """Detect outliers using IQR method.

    Args:
        values: Array of values
        multiplier: IQR multiplier (typically 1.5 or 3.0)

    Returns:
        Boolean array indicating outliers
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return (values < lower) | (values > upper)
