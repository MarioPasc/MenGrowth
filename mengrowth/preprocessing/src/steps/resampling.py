"""Resampling step to isotropic resolution.

Resamples MRI volumes to target voxel spacing using various methods
(bspline interpolation, ECLARE deep learning, composite).
"""

from typing import Dict, Any
import logging
from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_visualization_path,
    get_temp_path,
    log_step_start,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int
) -> Dict[str, Any]:
    """Execute resampling step.

    Args:
        context: Execution context with patient, study, modality info
        total_steps: Total number of per-modality steps in pipeline
        current_step_num: Current step number (1-indexed for logging)

    Returns:
        Dict with resampling results (spacing, shape info)

    Raises:
        RuntimeError: If resampling fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    paths = context.paths

    # Check if resampling is enabled
    method = config.resampling.method
    if method is None:
        logger.info(f"    [{current_step_num}/{total_steps}] Resampling skipped (method=None)")
        return {}

    log_step_start(logger, current_step_num, total_steps, context.step_name, method)

    # Get or create resampler component (lazy initialization with unique key per step instance)
    resampler = orchestrator._get_component(f"resampler_{method}_{context.step_name}", config)

    # Create temp file for resampled output
    temp_resampled = get_temp_path(context, operation="resampled")

    # Execute resampling on current state of the file
    result = resampler.execute(
        paths["nifti"],
        temp_resampled,
        allow_overwrite=True
    )

    # Visualization if enabled
    if config.save_visualization:
        viz_path = get_visualization_path(context)
        resampler.visualize(
            paths["nifti"],
            temp_resampled,
            viz_path,
            original_spacing=result["original_spacing"],
            target_spacing=result["target_spacing"],
            original_shape=result["original_shape"],
            resampled_shape=result["resampled_shape"]
        )

    # Replace original with resampled (in-place processing)
    temp_resampled.replace(paths["nifti"])

    logger.info(f"        - Resampling complete ({method})")

    return result
