"""Bias field correction step using N4 algorithm.

Removes intensity non-uniformity artifacts from MRI images.
"""

from typing import Dict, Any
import logging


from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_temp_path,
    get_visualization_path,
    log_step_start,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
) -> Dict[str, Any]:
    """Execute bias field correction step.

    Args:
        context: Execution context with patient, study, modality info
        total_steps: Total number of per-modality steps in pipeline
        current_step_num: Current step number (1-indexed for logging)

    Returns:
        Dict with bias field path and convergence data

    Raises:
        RuntimeError: If bias correction fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    paths = context.paths
    modality = context.modality

    # Check if bias correction is enabled
    method = config.bias_field_correction.method
    if method is None:
        logger.info(
            f"    [{current_step_num}/{total_steps}] Bias field correction skipped (method=None)"
        )
        return {}

    log_step_start(logger, current_step_num, total_steps, context.step_name, method)

    # Get or create bias corrector component
    bias_corrector = orchestrator._get_component(
        f"bias_corrector_{method}_{context.step_name}", config
    )

    # Create temp file for corrected output
    temp_corrected = get_temp_path(context, modality, "bias_corrected")

    # Determine where to save bias field artifact
    if config.save_artifact:
        bias_field_path = paths["bias_field"]
        logger.debug(f"    Saving bias field artifact to: {bias_field_path}")
    else:
        # Use temporary file that will be deleted
        bias_field_path = get_temp_path(context, modality, "bias_field")
        logger.debug("    Bias field artifact will not be saved (save_artifact=False)")

    # Execute bias correction
    result = bias_corrector.execute(
        paths["nifti"],
        temp_corrected,
        allow_overwrite=True,
        bias_field_output_path=bias_field_path,
    )

    # Clamp negative values to 0.  N4 divides by a B-spline bias field
    # estimate; at FOV edges where the field extrapolates poorly this can
    # produce negative intensities.  MRI magnitudes are physically
    # non-negative, so clamping is safe and prevents downstream issues
    # (negative fill in cubic padding, confused HD-BET inputs).
    import nibabel as nib
    import numpy as np

    corrected_img = nib.load(str(temp_corrected))
    corrected_data = corrected_img.get_fdata()
    neg_count = int((corrected_data < 0).sum())
    if neg_count > 0:
        neg_min = float(corrected_data.min())
        corrected_data = np.clip(corrected_data, 0, None)
        nib.Nifti1Image(
            corrected_data, corrected_img.affine, corrected_img.header
        ).to_filename(str(temp_corrected))
        logger.info(
            f"        - Clamped {neg_count} negative voxels to 0 "
            f"(min was {neg_min:.2f})"
        )

    # Visualization if enabled
    if config.save_visualization:
        viz_path = get_visualization_path(context)
        bias_corrector.visualize(
            paths["nifti"],
            temp_corrected,
            viz_path,
            bias_field_path=result["bias_field_path"],
            convergence_data=result["convergence_data"],
        )

    # Replace original with bias-corrected version (in-place processing)
    temp_corrected.replace(paths["nifti"])

    # Clean up temporary bias field if not saving
    if not config.save_artifact:
        if result["bias_field_path"].exists():
            result["bias_field_path"].unlink()
            logger.debug("    Temporary bias field artifact deleted")

    return result
