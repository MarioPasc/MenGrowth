"""Skull stripping step: brain extraction using HD-BET or SynthStrip.

This is a study-level step that processes all modalities.
"""

from typing import Dict, Any
from pathlib import Path
import tempfile
import logging
from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_output_dir,
    get_temp_path,
    get_artifact_path,
    get_visualization_path,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int
) -> Dict[str, Any]:
    """Execute skull stripping step (study-level operation).

    Args:
        context: Execution context (modality will be None for study-level steps)
        total_steps: Not used for study-level steps
        current_step_num: Not used for study-level steps

    Returns:
        Dict with skull stripping results per modality

    Raises:
        RuntimeError: If skull stripping fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    patient_id = context.patient_id
    study_dir = context.study_dir

    # Check if skull stripping is enabled
    method = config.skull_stripping.method
    if method is None:
        logger.info(f"\n  Executing study-level step: {context.step_name} - skipped (method=None)")
        return {}

    logger.info(f"\n  Executing study-level step: {context.step_name} ({method})")

    # Get or create skull stripper component
    skull_stripper = orchestrator._get_component(f"skull_stripper_{method}_{context.step_name}", config)

    # Determine output directories based on mode
    study_output_dir = get_output_dir(
        context=context
    )
    artifacts_base = Path(orchestrator.config.preprocessing_artifacts_path) / patient_id / study_dir.name
    viz_base = Path(orchestrator.config.viz_root) / patient_id / study_dir.name

    results = {}

    # Process each modality
    for modality in orchestrator.config.modalities:
        modality_path = study_output_dir / f"{modality}.nii.gz"

        # Skip if file doesn't exist
        if not modality_path.exists():
            logger.warning(f"  Skipping {modality} - file not found")
            continue

        logger.info(f"  Processing {modality}...")

        # Create temporary output path
        temp_skull_stripped = get_temp_path(context, modality, "skull_stripped")

        # Determine mask path
        if config.save_mask:
            mask_path = get_artifact_path(context, f"{modality}_brain_mask")
            logger.debug(f"    Saving brain mask to: {mask_path}")
        else:
            temp_dir = Path(tempfile.gettempdir())
            mask_path = temp_dir / f"_temp_{modality}_brain_mask.nii.gz"
            logger.debug("    Brain mask will not be saved (save_mask=False)")

        # Execute skull stripping
        result = skull_stripper.execute(
            modality_path,
            temp_skull_stripped,
            mask_path=mask_path,
            allow_overwrite=True
        )

        # Store results
        results[modality] = result

        # Generate visualization if enabled
        if config.save_visualization:
            viz_output = get_visualization_path(context, suffix=f"_{modality}")
            skull_stripper.visualize(
                modality_path,
                temp_skull_stripped,
                viz_output,
                **result
            )

        # Replace original with skull-stripped version (in-place)
        temp_skull_stripped.replace(modality_path)

        # Clean up temporary mask if not saving
        if not config.save_mask and mask_path.exists():
            mask_path.unlink()
            logger.debug("    Temporary brain mask deleted")

        logger.info(
            f"  {modality}: brain_volume={result['brain_volume_mm3']:.1f} mm³, "
            f"coverage={result['brain_coverage_percent']:.1f}%"
        )

    logger.info(f"  Skull stripping completed successfully")

    return results
