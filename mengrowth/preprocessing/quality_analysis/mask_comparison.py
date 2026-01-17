"""Mask comparison metrics for QC.

This module provides functions to compare brain masks:
- Independent masks (computed separately per timestamp via HD-BET)
- Propagated masks (warped from reference timestamp via registration transform)

The comparison helps assess:
1. Registration quality (high Dice = good alignment)
2. Temporal stability of skull stripping (consistent brain extraction)
3. Potential issues (low Dice may indicate registration failure or brain changes)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def compute_mask_comparison_metrics(
    independent_mask_path: Path,
    propagated_mask_path: Path,
    reference_mask_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Compute comparison metrics between independent and propagated masks.

    Args:
        independent_mask_path: Path to independently computed mask (e.g., HD-BET)
        propagated_mask_path: Path to propagated/warped mask from reference
        reference_mask_path: Optional path to reference mask for additional metrics

    Returns:
        Dict with comparison metrics:
            - dice_independent_vs_propagated: Dice coefficient
            - volume_independent_cc: Volume of independent mask in cc
            - volume_propagated_cc: Volume of propagated mask in cc
            - volume_difference_pct: Percentage volume difference
            - overlap_voxels: Number of overlapping voxels
            - independent_only_voxels: Voxels only in independent mask
            - propagated_only_voxels: Voxels only in propagated mask
    """
    metrics: Dict[str, Any] = {}

    try:
        # Load masks
        ind_mask = sitk.ReadImage(str(independent_mask_path))
        prop_mask = sitk.ReadImage(str(propagated_mask_path))

        # Convert to numpy arrays
        ind_arr = sitk.GetArrayFromImage(ind_mask).astype(bool)
        prop_arr = sitk.GetArrayFromImage(prop_mask).astype(bool)

        # Compute voxel volume
        spacing = ind_mask.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_cc = voxel_volume_mm3 / 1000.0

        # Volume metrics
        ind_volume_voxels = np.sum(ind_arr)
        prop_volume_voxels = np.sum(prop_arr)

        metrics["volume_independent_cc"] = float(ind_volume_voxels * voxel_volume_cc)
        metrics["volume_propagated_cc"] = float(prop_volume_voxels * voxel_volume_cc)

        # Volume difference percentage
        if ind_volume_voxels > 0:
            vol_diff_pct = abs(prop_volume_voxels - ind_volume_voxels) / ind_volume_voxels * 100
            metrics["volume_difference_pct"] = float(vol_diff_pct)
        else:
            metrics["volume_difference_pct"] = None

        # Overlap metrics
        intersection = np.sum(ind_arr & prop_arr)
        union = np.sum(ind_arr | prop_arr)
        ind_only = np.sum(ind_arr & ~prop_arr)
        prop_only = np.sum(prop_arr & ~ind_arr)

        metrics["overlap_voxels"] = int(intersection)
        metrics["independent_only_voxels"] = int(ind_only)
        metrics["propagated_only_voxels"] = int(prop_only)

        # Dice coefficient
        if (ind_volume_voxels + prop_volume_voxels) > 0:
            dice = 2 * intersection / (ind_volume_voxels + prop_volume_voxels)
            metrics["dice_independent_vs_propagated"] = float(dice)
        else:
            metrics["dice_independent_vs_propagated"] = 0.0

        # Jaccard index (IoU)
        if union > 0:
            jaccard = intersection / union
            metrics["jaccard_independent_vs_propagated"] = float(jaccard)
        else:
            metrics["jaccard_independent_vs_propagated"] = 0.0

        # Additional metrics with reference mask
        if reference_mask_path and reference_mask_path.exists():
            ref_mask = sitk.ReadImage(str(reference_mask_path))
            ref_arr = sitk.GetArrayFromImage(ref_mask).astype(bool)
            ref_volume_voxels = np.sum(ref_arr)

            metrics["volume_reference_cc"] = float(ref_volume_voxels * voxel_volume_cc)

            # Dice between reference and propagated (should be high if warping worked)
            if (ref_volume_voxels + prop_volume_voxels) > 0:
                dice_ref_prop = 2 * np.sum(ref_arr & prop_arr) / (ref_volume_voxels + prop_volume_voxels)
                metrics["dice_reference_vs_propagated"] = float(dice_ref_prop)

            # Dice between reference and independent (shows temporal stability)
            if (ref_volume_voxels + ind_volume_voxels) > 0:
                dice_ref_ind = 2 * np.sum(ref_arr & ind_arr) / (ref_volume_voxels + ind_volume_voxels)
                metrics["dice_reference_vs_independent"] = float(dice_ref_ind)

        metrics["status"] = "success"

    except Exception as e:
        logger.warning(f"Mask comparison failed: {e}")
        metrics["status"] = "error"
        metrics["error_message"] = str(e)

    return metrics


def compute_all_mask_comparisons(
    warped_masks_info: Dict[str, Dict[str, Path]],
    artifacts_dir: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """Compute mask comparisons for all timestamp/modality pairs.

    Args:
        warped_masks_info: Dict mapping "timestamp_modality" to {ref_mask, warped_mask}
        artifacts_dir: Base artifacts directory to find independent masks
        output_dir: Optional directory to save comparison results as JSON

    Returns:
        Dict mapping "timestamp_modality" to comparison metrics
    """
    all_comparisons: Dict[str, Dict[str, Any]] = {}

    for key, mask_paths in warped_masks_info.items():
        # Parse key to get timestamp and modality
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning(f"Unexpected key format: {key}")
            continue

        timestamp, modality = parts

        # Find independent mask path
        # Convention: artifacts_dir / study_name / {modality}_brain_mask.nii.gz
        # We need to find the study directory that ends with the timestamp
        ref_mask_path = mask_paths.get("ref_mask")
        warped_mask_path = mask_paths.get("warped_mask")

        if not ref_mask_path or not warped_mask_path:
            logger.warning(f"Missing mask paths for {key}")
            continue

        if not Path(warped_mask_path).exists():
            logger.warning(f"Warped mask not found: {warped_mask_path}")
            continue

        # The independent mask should be in the same location as the ref_mask
        # but for the moving (non-reference) timestamp
        # The warped_mask_path is in artifacts/longitudinal_registration/warped_masks/
        # The independent mask would be in artifacts/{study_name}/{modality}_brain_mask.nii.gz
        independent_mask_path = None

        # Try to find independent mask based on timestamp
        if artifacts_dir:
            for study_dir in artifacts_dir.iterdir():
                if study_dir.is_dir() and study_dir.name.endswith(f"-{timestamp}"):
                    candidate = study_dir / f"{modality}_brain_mask.nii.gz"
                    if candidate.exists():
                        independent_mask_path = candidate
                        break

        if independent_mask_path is None or not independent_mask_path.exists():
            logger.debug(f"Independent mask not found for {key}")
            continue

        # Compute comparison metrics
        logger.debug(f"Computing mask comparison for {key}")
        metrics = compute_mask_comparison_metrics(
            independent_mask_path=independent_mask_path,
            propagated_mask_path=Path(warped_mask_path),
            reference_mask_path=Path(ref_mask_path) if ref_mask_path else None
        )

        metrics["timestamp"] = timestamp
        metrics["modality"] = modality
        metrics["independent_mask_path"] = str(independent_mask_path)
        metrics["propagated_mask_path"] = str(warped_mask_path)

        all_comparisons[key] = metrics

    # Save to JSON if output_dir specified
    if output_dir and all_comparisons:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "mask_comparisons.json"
        with open(output_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2, default=str)
        logger.info(f"Saved mask comparisons to {output_file}")

    return all_comparisons


def summarize_mask_comparisons(comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize mask comparison metrics across all comparisons.

    Args:
        comparisons: Dict of comparison metrics from compute_all_mask_comparisons

    Returns:
        Summary statistics dict
    """
    if not comparisons:
        return {"n_comparisons": 0}

    dice_values = []
    volume_diffs = []

    for key, metrics in comparisons.items():
        if metrics.get("status") == "success":
            dice = metrics.get("dice_independent_vs_propagated")
            if dice is not None:
                dice_values.append(dice)

            vol_diff = metrics.get("volume_difference_pct")
            if vol_diff is not None:
                volume_diffs.append(vol_diff)

    summary = {
        "n_comparisons": len(comparisons),
        "n_successful": len(dice_values),
    }

    if dice_values:
        summary["dice_mean"] = float(np.mean(dice_values))
        summary["dice_std"] = float(np.std(dice_values))
        summary["dice_min"] = float(np.min(dice_values))
        summary["dice_max"] = float(np.max(dice_values))
        summary["dice_median"] = float(np.median(dice_values))

        # Flag potential issues (Dice < 0.85 may indicate problems)
        low_dice_count = sum(1 for d in dice_values if d < 0.85)
        summary["n_low_dice"] = low_dice_count
        summary["low_dice_fraction"] = low_dice_count / len(dice_values)

    if volume_diffs:
        summary["volume_diff_pct_mean"] = float(np.mean(volume_diffs))
        summary["volume_diff_pct_max"] = float(np.max(volume_diffs))

    return summary
