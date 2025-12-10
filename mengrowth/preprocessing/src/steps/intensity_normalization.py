"""Intensity normalization step.

Normalizes voxel intensities using various methods (zscore, KDE, percentile, etc.).
This step can be applied multiple times with different methods/parameters.
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
        logger.info(f"    [{current_step_num}/{total_steps}] Intensity normalization skipped (method=None)")
        return {}

    log_step_start(logger, current_step_num, total_steps, context.step_name, method)

    # Get or create normalizer component (lazy initialization with unique key per step instance)
    normalizer = orchestrator._get_component(f"normalizer_{method}_{context.step_name}", config)

    # Create temp file for normalized output
    temp_normalized = get_temp_path(context, operation="normalized")

    # Execute normalization
    result = normalizer.execute(
        paths["nifti"],
        temp_normalized,
        allow_overwrite=True
    )

    # Visualization if enabled
    if config.save_visualization:
        viz_path = get_visualization_path(context)
        normalizer.visualize(
            paths["nifti"],
            temp_normalized,
            viz_path,
            **result
        )

    # Replace original with normalized (in-place processing)
    temp_normalized.replace(paths["nifti"])

    logger.info(f"        - Normalization complete ({method})")

    return result
