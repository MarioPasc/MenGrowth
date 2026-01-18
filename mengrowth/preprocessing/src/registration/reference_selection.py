"""Automatic reference timestamp selection for longitudinal registration.

This module implements quality-based automatic selection of the reference
timestamp for longitudinal MRI registration. The selection prioritizes:

1. Primary: Image quality metrics (SNR, CNR, boundary sharpness)
2. Tiebreaker: Earlier timestamps (chronological ordering)
3. Validation: Jacobian determinant statistics to verify registration quality

Scientific rationale:
- Higher quality reference images produce more reliable registrations
- Earlier timepoints may have less disease progression (pre-treatment baseline)
- Jacobian determinant validation ensures physically plausible deformations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import yaml

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

logger = logging.getLogger(__name__)


@dataclass
class ReferenceSelectionConfig:
    """Configuration for automatic reference timestamp selection.

    Attributes:
        method: Selection method ("quality_based", "first", "last", "midpoint")
        quality_metrics: List of metrics to use for quality scoring
        quality_weights: Weights for each metric (defaults to equal weights)
        prefer_earlier: Use earlier timestamps as tiebreaker (default: True)
        min_quality_threshold: Minimum quality score threshold (0.0-1.0)
        validate_jacobian: Whether to compute Jacobian statistics for validation
        jacobian_log_threshold: Max allowed |log(det(J))| mean for valid registration
    """
    method: str = "quality_based"
    quality_metrics: List[str] = field(default_factory=lambda: [
        "snr_foreground",
        "cnr_high_low",
        "boundary_gradient_score"
    ])
    quality_weights: Optional[Dict[str, float]] = None
    prefer_earlier: bool = True
    min_quality_threshold: float = 0.0
    validate_jacobian: bool = True
    jacobian_log_threshold: float = 0.5  # Conservative threshold


class ReferenceSelector:
    """Selects optimal reference timestamp for longitudinal registration.

    This class implements a multi-criteria selection algorithm that:
    1. Computes quality scores for each available timestamp
    2. Ranks timestamps by composite quality score
    3. Uses chronological ordering as tiebreaker
    4. Optionally validates selection with Jacobian determinant analysis

    Attributes:
        config: Reference selection configuration
        verbose: Enable verbose logging
    """

    def __init__(
        self,
        config: ReferenceSelectionConfig,
        verbose: bool = False
    ) -> None:
        """Initialize reference selector.

        Args:
            config: Reference selection configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)

    def select_reference(
        self,
        study_dirs: List[Path],
        patient_id: str,
        modalities: List[str],
        qc_metrics_path: Optional[Path] = None,
        artifacts_base: Optional[Path] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Select optimal reference timestamp for a patient.

        Args:
            study_dirs: List of study directories for this patient
            patient_id: Patient identifier
            modalities: List of modalities to consider
            qc_metrics_path: Optional path to QC metrics CSV
            artifacts_base: Optional artifacts directory for saving selection info

        Returns:
            Tuple of (selected_timestamp, selection_info_dict)

        Raises:
            ValueError: If no valid timestamps found or selection fails
        """
        if len(study_dirs) < 1:
            raise ValueError(f"No study directories provided for {patient_id}")

        if len(study_dirs) == 1:
            timestamp = self._extract_timestamp(study_dirs[0])
            self._logger.info(f"Only one timestamp available: {timestamp}")
            return timestamp, {"method": "single_available", "timestamp": timestamp}

        # Extract timestamps from study directories
        timestamps = [self._extract_timestamp(d) for d in study_dirs]

        # Apply selection method
        if self.config.method == "first":
            selected = self._select_first(timestamps)
            method = "first"
        elif self.config.method == "last":
            selected = self._select_last(timestamps)
            method = "last"
        elif self.config.method == "midpoint":
            selected = self._select_midpoint(timestamps)
            method = "midpoint"
        elif self.config.method == "quality_based":
            selected, selection_info = self._select_quality_based(
                study_dirs=study_dirs,
                timestamps=timestamps,
                modalities=modalities,
                qc_metrics_path=qc_metrics_path
            )
            method = "quality_based"

            # Save selection info if artifacts directory provided
            if artifacts_base:
                self._save_selection_info(
                    selection_info=selection_info,
                    patient_id=patient_id,
                    artifacts_base=artifacts_base
                )

            return selected, selection_info
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")

        return selected, {"method": method, "timestamp": selected}

    def _extract_timestamp(self, study_dir: Path) -> str:
        """Extract timestamp from study directory name.

        Expects format: PatientID-XXX (e.g., MenGrowth-0006-001 -> 001)

        Args:
            study_dir: Study directory path

        Returns:
            Timestamp string (e.g., "001")
        """
        return study_dir.name.split('-')[-1]

    def _select_first(self, timestamps: List[str]) -> str:
        """Select the first (earliest) timestamp."""
        return min(timestamps)

    def _select_last(self, timestamps: List[str]) -> str:
        """Select the last (latest) timestamp."""
        return max(timestamps)

    def _select_midpoint(self, timestamps: List[str]) -> str:
        """Select the middle timestamp (chronological)."""
        sorted_ts = sorted(timestamps)
        midpoint_idx = len(sorted_ts) // 2
        return sorted_ts[midpoint_idx]

    def _select_quality_based(
        self,
        study_dirs: List[Path],
        timestamps: List[str],
        modalities: List[str],
        qc_metrics_path: Optional[Path]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select timestamp based on image quality metrics.

        Priority:
        1. Composite quality score (SNR, CNR, boundary sharpness)
        2. Earlier timestamps as tiebreaker

        Args:
            study_dirs: Study directories
            timestamps: Corresponding timestamps
            modalities: Modalities to evaluate
            qc_metrics_path: Optional path to pre-computed QC metrics

        Returns:
            Tuple of (selected_timestamp, detailed_selection_info)
        """
        selection_info: Dict[str, Any] = {
            "method": "quality_based",
            "timestamps_evaluated": timestamps,
            "quality_scores": {},
            "metrics_per_timestamp": {},
            "ranking": [],
        }

        # Try to load pre-computed QC metrics
        qc_metrics = None
        if qc_metrics_path and qc_metrics_path.exists():
            try:
                qc_metrics = self._load_qc_metrics(qc_metrics_path)
                self._logger.info(f"Loaded QC metrics from {qc_metrics_path}")
            except Exception as e:
                self._logger.warning(f"Failed to load QC metrics: {e}")

        # Compute quality scores for each timestamp
        quality_scores: Dict[str, float] = {}
        metrics_per_ts: Dict[str, Dict[str, float]] = {}

        for study_dir, timestamp in zip(study_dirs, timestamps):
            metrics = self._compute_quality_metrics(
                study_dir=study_dir,
                modalities=modalities,
                qc_metrics=qc_metrics,
                timestamp=timestamp
            )
            metrics_per_ts[timestamp] = metrics

            # Compute composite score
            score = self._compute_composite_score(metrics)
            quality_scores[timestamp] = score

            if self.verbose:
                self._logger.debug(
                    f"Timestamp {timestamp}: score={score:.4f}, metrics={metrics}"
                )

        selection_info["quality_scores"] = quality_scores
        selection_info["metrics_per_timestamp"] = metrics_per_ts

        # Rank by quality (descending), then by timestamp (ascending for earlier preference)
        if self.config.prefer_earlier:
            # Sort by score descending, then timestamp ascending
            ranking = sorted(
                quality_scores.items(),
                key=lambda x: (-x[1], x[0])
            )
        else:
            # Sort by score descending only
            ranking = sorted(
                quality_scores.items(),
                key=lambda x: -x[1]
            )

        selection_info["ranking"] = [(ts, score) for ts, score in ranking]

        # Select top-ranked timestamp
        selected_timestamp = ranking[0][0]
        selection_info["selected_timestamp"] = selected_timestamp
        selection_info["selected_score"] = ranking[0][1]

        # Check minimum quality threshold
        if ranking[0][1] < self.config.min_quality_threshold:
            self._logger.warning(
                f"Best quality score {ranking[0][1]:.4f} is below threshold "
                f"{self.config.min_quality_threshold}. Using earliest timestamp as fallback."
            )
            selected_timestamp = self._select_first(timestamps)
            selection_info["fallback_used"] = True
            selection_info["fallback_reason"] = "below_quality_threshold"

        self._logger.info(
            f"Selected reference timestamp: {selected_timestamp} "
            f"(score={quality_scores[selected_timestamp]:.4f})"
        )

        return selected_timestamp, selection_info

    def _load_qc_metrics(self, qc_path: Path) -> Dict[str, Any]:
        """Load pre-computed QC metrics from CSV.

        Args:
            qc_path: Path to QC metrics CSV (wide or long format)

        Returns:
            Dict mapping study_id -> metric_name -> value
        """
        import pandas as pd

        df = pd.read_csv(qc_path)

        # Detect format and parse
        metrics_dict: Dict[str, Dict[str, float]] = {}

        # Check if it's wide format (one row per image)
        if "study_id" in df.columns:
            for _, row in df.iterrows():
                study_id = row.get("study_id", "")
                if study_id:
                    metrics_dict[study_id] = {}
                    for col in df.columns:
                        if col not in ["patient_id", "study_id", "modality", "step_name"]:
                            try:
                                val = float(row[col])
                                if not np.isnan(val):
                                    metrics_dict[study_id][col] = val
                            except (ValueError, TypeError):
                                pass

        return metrics_dict

    def _compute_quality_metrics(
        self,
        study_dir: Path,
        modalities: List[str],
        qc_metrics: Optional[Dict[str, Any]],
        timestamp: str
    ) -> Dict[str, float]:
        """Compute quality metrics for a study.

        Tries to use pre-computed QC metrics first, falls back to
        computing from images if necessary.

        Args:
            study_dir: Study directory path
            modalities: Modalities to evaluate
            qc_metrics: Pre-computed QC metrics dict
            timestamp: Study timestamp

        Returns:
            Dict of metric_name -> value
        """
        metrics: Dict[str, float] = {}

        # Try pre-computed metrics first
        study_id = study_dir.name
        if qc_metrics and study_id in qc_metrics:
            precomputed = qc_metrics[study_id]
            for metric_name in self.config.quality_metrics:
                if metric_name in precomputed:
                    metrics[metric_name] = precomputed[metric_name]

            # If all metrics found, return
            if len(metrics) == len(self.config.quality_metrics):
                return metrics

        # Compute missing metrics from images
        for modality in modalities:
            image_path = study_dir / f"{modality}.nii.gz"
            if not image_path.exists():
                continue

            try:
                computed = self._compute_metrics_from_image(image_path)
                for metric_name in self.config.quality_metrics:
                    if metric_name not in metrics and metric_name in computed:
                        metrics[metric_name] = computed[metric_name]
            except Exception as e:
                self._logger.warning(
                    f"Failed to compute metrics for {image_path}: {e}"
                )
                continue

            # Stop if we have all metrics
            if len(metrics) >= len(self.config.quality_metrics):
                break

        return metrics

    def _compute_metrics_from_image(self, image_path: Path) -> Dict[str, float]:
        """Compute quality metrics directly from image.

        Args:
            image_path: Path to NIfTI image

        Returns:
            Dict of computed metrics
        """
        image = sitk.ReadImage(str(image_path))
        arr = sitk.GetArrayFromImage(image)

        metrics: Dict[str, float] = {}

        # Mask out background (assume zeros or very low values are background)
        nonzero_mask = arr > 0
        if not np.any(nonzero_mask):
            return metrics

        foreground = arr[nonzero_mask]

        # SNR foreground (Kaufman method approximation)
        if len(foreground) > 100:
            # Use percentiles to estimate signal region
            p25 = np.percentile(foreground, 25)
            p75 = np.percentile(foreground, 75)
            signal_region = foreground[(foreground >= p25) & (foreground <= p75)]

            if len(signal_region) > 10:
                snr_foreground = np.mean(signal_region) / (np.std(signal_region) + 1e-8)
                # Apply correction factor for Rician noise
                correction_factor = np.sqrt(2.0 / (4.0 - np.pi))
                metrics["snr_foreground"] = float(snr_foreground * correction_factor)

        # CNR high-low (contrast between intensity regions)
        if len(foreground) > 100:
            p_low = np.percentile(foreground, 25)
            p_mid = np.percentile(foreground, 50)
            p_high = np.percentile(foreground, 75)

            low_region = foreground[foreground < p_mid]
            high_region = foreground[foreground >= p_mid]

            if len(low_region) > 10 and len(high_region) > 10:
                mean_low = np.mean(low_region)
                mean_high = np.mean(high_region)
                var_low = np.var(low_region)
                var_high = np.var(high_region)
                pooled_std = np.sqrt(var_low + var_high)

                if pooled_std > 1e-8:
                    cnr = abs(mean_high - mean_low) / pooled_std
                    metrics["cnr_high_low"] = float(cnr)

        # Boundary gradient score (sharpness at edges)
        try:
            gradient_score = self._compute_boundary_gradient_score(arr, nonzero_mask)
            metrics["boundary_gradient_score"] = gradient_score
        except Exception:
            pass

        return metrics

    def _compute_boundary_gradient_score(
        self,
        arr: np.ndarray,
        foreground_mask: np.ndarray
    ) -> float:
        """Compute boundary gradient score (edge sharpness).

        Higher values indicate sharper boundaries, suggesting
        less motion blur and better image quality.

        Args:
            arr: Image array
            foreground_mask: Boolean mask of foreground

        Returns:
            Boundary gradient score (higher is better)
        """
        from scipy import ndimage

        # Erode mask to find interior
        eroded = binary_erosion(foreground_mask, iterations=2)

        # Boundary region = original mask minus eroded
        boundary = foreground_mask & ~eroded

        if not np.any(boundary):
            return 0.0

        # Compute gradient magnitude
        grad_x = ndimage.sobel(arr.astype(np.float64), axis=0)
        grad_y = ndimage.sobel(arr.astype(np.float64), axis=1)
        grad_z = ndimage.sobel(arr.astype(np.float64), axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # Mean gradient at boundary
        boundary_gradient = gradient_magnitude[boundary]

        if len(boundary_gradient) == 0:
            return 0.0

        return float(np.mean(boundary_gradient))

    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite quality score from individual metrics.

        Uses weighted combination with z-score normalization within
        the set of candidates.

        Args:
            metrics: Dict of metric_name -> value

        Returns:
            Composite score (higher is better)
        """
        if not metrics:
            return 0.0

        # Get weights (default to equal weights)
        weights = self.config.quality_weights or {}

        # Compute weighted sum
        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name in self.config.quality_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                weight = weights.get(metric_name, 1.0)

                # Normalize metric to [0, 1] range approximately
                # Using sigmoid-like transformation
                normalized = self._normalize_metric(metric_name, value)

                weighted_sum += weight * normalized
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric value to approximately [0, 1] range.

        Uses metric-specific scaling based on typical ranges.

        Args:
            metric_name: Name of the metric
            value: Raw metric value

        Returns:
            Normalized value in [0, 1]
        """
        # Typical ranges for brain MRI metrics
        ranges = {
            "snr_foreground": (5.0, 50.0),      # Typical SNR range
            "snr_background": (5.0, 100.0),
            "cnr_high_low": (0.5, 5.0),         # Typical CNR range
            "boundary_gradient_score": (10.0, 500.0),  # Gradient magnitude
        }

        if metric_name in ranges:
            min_val, max_val = ranges[metric_name]
            normalized = (value - min_val) / (max_val - min_val)
            return float(np.clip(normalized, 0.0, 1.0))

        # Default: assume positive metric, use log scaling
        if value <= 0:
            return 0.0
        return float(np.clip(np.log1p(value) / 5.0, 0.0, 1.0))

    def _save_selection_info(
        self,
        selection_info: Dict[str, Any],
        patient_id: str,
        artifacts_base: Path
    ) -> None:
        """Save selection information to artifacts directory.

        Args:
            selection_info: Selection details dict
            patient_id: Patient identifier
            artifacts_base: Base artifacts directory
        """
        output_dir = artifacts_base / "longitudinal_registration"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "reference_selection_info.json"

        # Convert numpy types to Python types
        def convert_types(obj: Any) -> Any:
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        info_to_save = {
            "patient_id": patient_id,
            **convert_types(selection_info)
        }

        with open(output_file, 'w') as f:
            json.dump(info_to_save, f, indent=2)

        self._logger.info(f"Saved reference selection info to {output_file}")


def compute_jacobian_statistics(
    transform_path: Path,
    reference_image_path: Path
) -> Dict[str, float]:
    """Compute Jacobian determinant statistics for registration validation.

    The Jacobian determinant indicates local volume change:
    - det(J) = 1: No volume change
    - det(J) > 1: Local expansion
    - det(J) < 1: Local contraction
    - det(J) < 0: Folding (physically implausible)

    The log of the Jacobian determinant is often used:
    - log(det(J)) = 0: No change
    - |log(det(J))| large: Large deformation

    For affine transforms (Rigid, Affine), the Jacobian is constant and equals
    the determinant of the linear transformation matrix.

    Args:
        transform_path: Path to transform file (.h5, .mat, or composite)
        reference_image_path: Path to reference image (defines output space)

    Returns:
        Dict with Jacobian statistics:
        - jacobian_det_mean: Mean Jacobian determinant
        - jacobian_det_std: Std of Jacobian determinant
        - jacobian_log_det_mean: Mean of log(det(J))
        - jacobian_log_det_std: Std of log(det(J))
        - jacobian_negative_fraction: Fraction of negative determinants (folding)
        - jacobian_min: Minimum determinant
        - jacobian_max: Maximum determinant
        - transform_type: Type of transform detected
    """
    import ants

    transform_path = Path(transform_path)

    if not transform_path.exists():
        return {
            "jacobian_det_mean": np.nan,
            "jacobian_det_std": np.nan,
            "jacobian_log_det_mean": np.nan,
            "jacobian_log_det_std": np.nan,
            "jacobian_negative_fraction": np.nan,
            "jacobian_min": np.nan,
            "jacobian_max": np.nan,
            "error": f"Transform file not found: {transform_path}"
        }

    try:
        # Try to read the transform using ANTsPy
        tx = ants.read_transform(str(transform_path))
        tx_type = tx.transform_type

        logger.debug(f"Transform type: {tx_type}")

        # For affine transforms, compute Jacobian determinant from matrix
        # Affine transforms have constant Jacobian (determinant of linear part)
        if "Affine" in tx_type or "Rigid" in tx_type or "MatrixOffsetTransformBase" in tx_type:
            # Get the affine parameters
            params = tx.parameters

            # For 3D affine: 12 parameters (9 matrix + 3 translation)
            # Matrix is stored row-major: [a00, a01, a02, a10, a11, a12, a20, a21, a22, t0, t1, t2]
            if len(params) >= 9:
                # Extract 3x3 linear part
                matrix = np.array([
                    [params[0], params[1], params[2]],
                    [params[3], params[4], params[5]],
                    [params[6], params[7], params[8]]
                ])

                # Compute determinant
                jacobian_det = np.linalg.det(matrix)
                jacobian_log_det = np.log(abs(jacobian_det)) if jacobian_det > 0 else np.nan

                stats = {
                    "jacobian_det_mean": float(jacobian_det),
                    "jacobian_det_std": 0.0,  # Constant for affine
                    "jacobian_log_det_mean": float(jacobian_log_det) if not np.isnan(jacobian_log_det) else 0.0,
                    "jacobian_log_det_std": 0.0,  # Constant for affine
                    "jacobian_negative_fraction": 1.0 if jacobian_det < 0 else 0.0,
                    "jacobian_min": float(jacobian_det),
                    "jacobian_max": float(jacobian_det),
                    "transform_type": "affine",
                    "matrix_determinant": float(jacobian_det)
                }

                logger.debug(f"Affine Jacobian determinant: {jacobian_det:.6f}")
                return stats

        # For deformable transforms, try to compute Jacobian image
        # Load reference image for output space
        ref_img = ants.image_read(str(reference_image_path))

        try:
            # Compute Jacobian image
            jacobian_img = ants.create_jacobian_determinant_image(
                domain_image=ref_img,
                tx=[str(transform_path)],
                do_log=False
            )
            jacobian_arr = jacobian_img.numpy()

            # Compute log Jacobian
            jacobian_log_img = ants.create_jacobian_determinant_image(
                domain_image=ref_img,
                tx=[str(transform_path)],
                do_log=True
            )
            jacobian_log_arr = jacobian_log_img.numpy()

            # Filter out background (where Jacobian is exactly 0 or 1)
            valid_mask = (jacobian_arr != 0) & (jacobian_arr != 1)

            if np.any(valid_mask):
                jac_valid = jacobian_arr[valid_mask]
                jac_log_valid = jacobian_log_arr[valid_mask]
            else:
                jac_valid = jacobian_arr.flatten()
                jac_log_valid = jacobian_log_arr.flatten()

            stats = {
                "jacobian_det_mean": float(np.mean(jac_valid)),
                "jacobian_det_std": float(np.std(jac_valid)),
                "jacobian_log_det_mean": float(np.mean(jac_log_valid)),
                "jacobian_log_det_std": float(np.std(jac_log_valid)),
                "jacobian_negative_fraction": float(np.mean(jac_valid < 0)),
                "jacobian_min": float(np.min(jac_valid)),
                "jacobian_max": float(np.max(jac_valid)),
                "transform_type": "deformable"
            }

            return stats

        except Exception as jac_err:
            # If Jacobian image computation fails, return partial stats from transform
            logger.debug(f"Jacobian image computation failed: {jac_err}")
            return {
                "jacobian_det_mean": 1.0,  # Assume identity-like
                "jacobian_det_std": 0.0,
                "jacobian_log_det_mean": 0.0,
                "jacobian_log_det_std": 0.0,
                "jacobian_negative_fraction": 0.0,
                "jacobian_min": 1.0,
                "jacobian_max": 1.0,
                "transform_type": tx_type,
                "note": "Jacobian image computation failed, using default values"
            }

    except Exception as e:
        logger.warning(f"Failed to compute Jacobian statistics: {e}")
        return {
            "jacobian_det_mean": np.nan,
            "jacobian_det_std": np.nan,
            "jacobian_log_det_mean": np.nan,
            "jacobian_log_det_std": np.nan,
            "jacobian_negative_fraction": np.nan,
            "jacobian_min": np.nan,
            "jacobian_max": np.nan,
            "error": str(e)
        }


def validate_registration_quality(
    jacobian_stats: Dict[str, float],
    config: ReferenceSelectionConfig
) -> Tuple[bool, str]:
    """Validate registration quality using Jacobian statistics.

    Args:
        jacobian_stats: Jacobian statistics dict
        config: Reference selection config with thresholds

    Returns:
        Tuple of (is_valid, message)
    """
    if not config.validate_jacobian:
        return True, "Jacobian validation disabled"

    # Check for computation errors
    if "error" in jacobian_stats:
        return False, f"Jacobian computation failed: {jacobian_stats['error']}"

    # Check for folding (negative determinants)
    neg_frac = jacobian_stats.get("jacobian_negative_fraction", 0.0)
    if neg_frac > 0.01:  # More than 1% folding is problematic
        return False, f"Excessive folding detected: {neg_frac*100:.2f}% negative Jacobian"

    # Check deformation magnitude
    log_mean = abs(jacobian_stats.get("jacobian_log_det_mean", 0.0))
    if log_mean > config.jacobian_log_threshold:
        return False, (
            f"Large mean deformation: |log(det(J))|={log_mean:.4f} "
            f"> threshold {config.jacobian_log_threshold}"
        )

    # Check for extreme local deformations
    jac_min = jacobian_stats.get("jacobian_min", 1.0)
    jac_max = jacobian_stats.get("jacobian_max", 1.0)

    if jac_min < 0.1:  # More than 90% local contraction
        return False, f"Extreme local contraction: min(det(J))={jac_min:.4f}"

    if jac_max > 10.0:  # More than 10x local expansion
        return False, f"Extreme local expansion: max(det(J))={jac_max:.4f}"

    return True, "Registration quality validated"
