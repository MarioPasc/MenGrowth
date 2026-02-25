"""Intensity normalization step.

Normalizes voxel intensities using various methods (zscore, KDE, percentile, etc.).
This step can be applied multiple times with different methods/parameters.

When method=None and save_visualization=True, generates a passthrough visualization
showing the current image state (3 views + histogram) without modifying the data.
This is useful for QC when normalization will be done on-the-fly downstream.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_visualization_path,
    get_temp_path,
    log_step_start,
)

logger = logging.getLogger(__name__)


def _generate_passthrough_visualization(nifti_path: Path, viz_path: Path) -> None:
    """Generate a snapshot visualization of the current image without normalization.

    Shows axial, sagittal, coronal views and an intensity histogram so that the
    image state at this pipeline position can be inspected for QC even when
    normalization is disabled (method=None).

    Args:
        nifti_path: Path to the current NIfTI file.
        viz_path: Path where the visualization PNG will be saved.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()

        mid_z = data.shape[2] // 2
        mid_x = data.shape[0] // 2
        mid_y = data.shape[1] // 2

        axial = data[:, :, mid_z].T
        sagittal = data[mid_x, :, :].T
        coronal = data[:, mid_y, :].T

        vmin, vmax = float(data.min()), float(data.max())

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"Intensity Snapshot (no normalization): {nifti_path.stem}",
            fontsize=14,
            fontweight="bold",
        )

        for ax, slc, title in zip(
            axes[:3],
            [axial, sagittal, coronal],
            ["Axial", "Sagittal", "Coronal"],
        ):
            ax.imshow(slc, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        # Histogram of non-zero voxels
        nonzero = data[data > 0].ravel()
        if nonzero.size > 0:
            mean_val = float(np.mean(nonzero))
            std_val = float(np.std(nonzero))
            axes[3].hist(nonzero, bins=100, alpha=0.7, color="blue", density=True)
            axes[3].axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={mean_val:.1f}",
            )
            axes[3].axvline(
                mean_val - std_val,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="μ±σ",
            )
            axes[3].axvline(
                mean_val + std_val,
                color="orange",
                linestyle=":",
                linewidth=1.5,
            )
            axes[3].legend(fontsize=8)
        axes[3].set_title("Intensity Histogram", fontsize=12)
        axes[3].set_xlabel("Intensity", fontsize=10)
        axes[3].set_ylabel("Density", fontsize=10)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"        - Passthrough visualization saved: {viz_path}")

    except Exception as e:
        logger.warning(f"        - Failed to generate passthrough visualization: {e}")


def _resolve_brain_mask_path(context: StepExecutionContext) -> Optional[Path]:
    """Locate the skull-stripping brain mask from the artifacts directory.

    Looks for: {artifacts_base}/{patient_id}/{study_dir}/{modality}_brain_mask.nii.gz

    Args:
        context: Execution context with patient, study, modality info

    Returns:
        Path to brain mask if found, None otherwise
    """
    try:
        artifacts_base = Path(context.orchestrator.config.preprocessing_artifacts_path)
        study_name = context.study_dir.name
        modality = context.modality

        mask_path = (
            artifacts_base
            / context.patient_id
            / study_name
            / f"{modality}_brain_mask.nii.gz"
        )

        if mask_path.exists():
            logger.debug(f"        - Found brain mask: {mask_path}")
            return mask_path
        else:
            logger.debug(
                f"        - No brain mask found at {mask_path}, normalizer will use nonzero fallback"
            )
            return None
    except Exception as e:
        logger.debug(f"        - Could not resolve brain mask path: {e}")
        return None


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
) -> Dict[str, Any]:
    """Execute intensity normalization step.

    Args:
        context: Execution context with patient, study, modality info
        total_steps: Total number of per-modality steps in pipeline
        current_step_num: Current step number (1-indexed for logging)

    Returns:
        Dict with normalization results (method-specific metadata)

    Raises:
        RuntimeError: If normalization fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    paths = context.paths
    modality = context.modality

    # Check if normalization is enabled
    method = config.intensity_normalization.method
    if method is None:
        logger.info(
            f"    [{current_step_num}/{total_steps}] Intensity normalization skipped (method=None)"
        )
        # Still generate passthrough visualization for QC
        if config.save_visualization:
            viz_path = get_visualization_path(context)
            _generate_passthrough_visualization(paths["nifti"], viz_path)
        return {}

    log_step_start(logger, current_step_num, total_steps, context.step_name, method)

    # Get or create normalizer component (lazy initialization with unique key per step instance)
    normalizer = orchestrator._get_component(
        f"normalizer_{method}_{context.step_name}", config
    )

    # Create temp file for normalized output
    temp_normalized = get_temp_path(context, operation="normalized")

    # Resolve brain mask from skull stripping artifacts
    mask_path = _resolve_brain_mask_path(context)

    # Execute normalization with brain mask
    result = normalizer.execute(
        paths["nifti"], temp_normalized, allow_overwrite=True, mask_path=mask_path
    )

    # Log brain coverage metrics if available
    if result.get("brain_coverage_percent") is not None:
        logger.info(
            f"        - Brain mask: {result.get('mask_source', 'unknown')}, "
            f"coverage: {result['brain_coverage_percent']:.1f}%, "
            f"voxels: {result.get('brain_voxel_count', 'N/A')}"
        )

    # Visualization if enabled
    if config.save_visualization:
        viz_path = get_visualization_path(context)
        normalizer.visualize(paths["nifti"], temp_normalized, viz_path, **result)

    # Replace original with normalized (in-place processing)
    temp_normalized.replace(paths["nifti"])

    logger.info(f"        - Normalization complete ({method})")

    return result
