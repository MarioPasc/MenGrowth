"""Data harmonization step: NRRD→NIfTI conversion, reorientation, background removal.

This is a composite step that performs three sub-operations sequentially:
1. NRRD to NIfTI conversion
2. Reorientation to specified orientation (e.g., RAS)
3. Background removal (conditional)
"""

from typing import Dict, Any
import logging
from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_temp_path,
    log_step_start,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int
) -> Dict[str, Any]:
    """Execute data harmonization step.

    Args:
        context: Execution context with patient, study, modality info
        total_steps: Total number of per-modality steps in pipeline
        current_step_num: Current step number (1-indexed for logging)

    Returns:
        Dict with execution results (empty for this step)

    Raises:
        RuntimeError: If any sub-step fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    paths = context.paths
    modality = context.modality

    log_step_start(logger, current_step_num, total_steps, context.step_name)

    results = {}

    # Sub-step 1: NRRD to NIfTI conversion
    logger.info(f"        - Converting NRRD to NIfTI...")
    converter = orchestrator._get_component("converter", config)

    # Find input NRRD file
    input_file = context.study_dir / f"{modality}.nrrd"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    converter.execute(
        input_file,
        paths["nifti"],
        allow_overwrite=orchestrator.config.overwrite
    )

    if config.save_visualization:
        converter.visualize(input_file, paths["nifti"], paths["viz_convert"])

    # Sub-step 2: Reorientation
    logger.info(f"        - Reorienting to {config.reorient_to}...")
    reorienter = orchestrator._get_component("reorienter", config)
    temp_reoriented = get_temp_path(context, modality, "reoriented")

    reorienter.execute(paths["nifti"], temp_reoriented, allow_overwrite=True)

    if config.save_visualization:
        reorienter.visualize(paths["nifti"], temp_reoriented, paths["viz_reorient"])

    # Replace original with reoriented
    temp_reoriented.replace(paths["nifti"])

    # Sub-step 3: Background removal (if enabled)
    bg_method = config.background_zeroing.method
    if bg_method is not None:
        logger.info(f"        - Removing background ({bg_method})...")
        bg_remover = orchestrator._get_component(f"background_remover_{bg_method}", config)
        temp_masked = get_temp_path(context, modality, "masked")

        bg_remover.execute(paths["nifti"], temp_masked, allow_overwrite=True)

        if config.save_visualization:
            bg_remover.visualize(paths["nifti"], temp_masked, paths["viz_background"])

        # Replace with masked version
        temp_masked.replace(paths["nifti"])
    else:
        logger.info(f"        - Background removal skipped (method=None)")

    return results
