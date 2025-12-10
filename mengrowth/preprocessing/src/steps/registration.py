"""Registration step: intra-study coregistration and atlas registration.

This is a study-level step that operates on all modalities together:
1. Intra-study multi-modal coregistration to reference modality
2. Intra-study to atlas registration (optional)
"""

from typing import Dict, Any
from pathlib import Path
import logging
from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_output_dir,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int
) -> Dict[str, Any]:
    """Execute registration step (study-level operation).

    Args:
        context: Execution context (modality will be None for study-level steps)
        total_steps: Not used for study-level steps
        current_step_num: Not used for study-level steps

    Returns:
        Dict with registration results (reference modality, transforms)

    Raises:
        RuntimeError: If registration fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    patient_id = context.patient_id
    study_dir = context.study_dir

    logger.info(f"\n  Executing study-level step: {context.step_name}")

    results = {
        "reference_modality": None,
        "transforms": {},
        "registered_modalities": []
    }

    # Determine study output directory based on mode
    study_output_dir = get_output_dir(
        context=context
    )
    artifacts_base = Path(orchestrator.config.preprocessing_artifacts_path) / patient_id / study_dir.name
    viz_base = Path(orchestrator.config.viz_root) / patient_id / study_dir.name

    # Step 3a: Intra-study multi-modal coregistration to reference
    intra_study_transforms = {}
    intra_study_method = config.intra_study_to_reference.method if hasattr(config, 'intra_study_to_reference') else None

    if intra_study_method is not None:
        logger.info(f"  [Sub-step 3a] Intra-study multi-modal coregistration ({intra_study_method})")

        # Get or create intra-study registrator
        intra_study_registrator = orchestrator._get_component(
            f"intra_study_to_ref_{context.step_name}",
            config
        )

        try:
            # Execute intra-study to reference registration
            reg_result = intra_study_registrator.execute(
                study_dir=study_output_dir,
                artifacts_dir=artifacts_base,
                modalities=orchestrator.config.modalities
            )

            # Store results
            results["reference_modality"] = reg_result["reference_modality"]
            results["transforms"] = reg_result["transforms"]
            results["registered_modalities"] = reg_result.get("registered_modalities", [])
            intra_study_transforms = reg_result["transforms"]

            # Update orchestrator state
            orchestrator.selected_reference_modality = reg_result["reference_modality"]

            logger.info(f"  Reference modality: {results['reference_modality']}")

            # Generate visualizations if enabled
            if config.save_visualization:
                reference_path = study_output_dir / f"{results['reference_modality']}.nii.gz"

                for modality in reg_result["registered_modalities"]:
                    moving_path = study_output_dir / f"{modality}.nii.gz"
                    transform_path = reg_result["transforms"].get(modality)

                    viz_output = viz_base / f"{context.step_name}_3a_intra_study_{modality}_to_{results['reference_modality']}.png"

                    try:
                        intra_study_registrator.visualize(
                            reference_path=reference_path,
                            moving_path=moving_path,
                            registered_path=moving_path,  # Already replaced
                            output_path=viz_output,
                            modality=modality,
                            transform_path=transform_path
                        )
                    except Exception as viz_error:
                        logger.warning(f"  Failed to generate visualization for {modality}: {viz_error}")

            logger.info(f"  Successfully registered {len(reg_result['registered_modalities'])} modalities to reference")

        except Exception as e:
            logger.error(f"  [Error] Intra-study to reference registration failed: {e}")
            raise RuntimeError(f"Intra-study registration failed") from e
    else:
        logger.info("  [Sub-step 3a] Intra-study to reference registration skipped (method=None)")

    # Step 3b: Intra-study to atlas registration
    atlas_method = config.intra_study_to_atlas.method if hasattr(config, 'intra_study_to_atlas') else None

    if atlas_method is not None and results["reference_modality"] is not None:
        logger.info(f"  [Sub-step 3b] Intra-study to atlas registration ({atlas_method})")

        # Get or create atlas registrator
        atlas_registrator = orchestrator._get_component(
            f"intra_study_to_atlas_{context.step_name}",
            config
        )

        try:
            # Execute atlas registration
            atlas_result = atlas_registrator.execute(
                study_dir=study_output_dir,
                artifacts_dir=artifacts_base,
                modalities=orchestrator.config.modalities,
                intra_study_transforms=intra_study_transforms
            )

            logger.info(f"  Reference registered to atlas: {atlas_result['atlas_path']}")

            # Generate visualizations if enabled
            if config.save_visualization:
                atlas_path = Path(config.intra_study_to_atlas.atlas_path)
                reference_path = study_output_dir / f"{results['reference_modality']}.nii.gz"

                # Visualize reference to atlas
                viz_output_ref = viz_base / f"{context.step_name}_3b_atlas_ref_{results['reference_modality']}.png"
                try:
                    atlas_registrator.visualize_reference_to_atlas(
                        atlas_path=atlas_path,
                        reference_path=reference_path,
                        output_path=viz_output_ref,
                        ref_to_atlas_transform=atlas_result["ref_to_atlas_transform"]
                    )
                except Exception as viz_error:
                    logger.warning(f"  Failed to generate reference→atlas visualization: {viz_error}")

                # Visualize each modality in atlas space
                for modality in atlas_result["registered_modalities"]:
                    modality_path = study_output_dir / f"{modality}.nii.gz"
                    viz_output = viz_base / f"{context.step_name}_3b_atlas_{modality}.png"

                    try:
                        atlas_registrator.visualize_modality_in_atlas_space(
                            atlas_path=atlas_path,
                            modality_path=modality_path,
                            output_path=viz_output,
                            modality=modality
                        )
                    except Exception as viz_error:
                        logger.warning(f"  Failed to generate atlas visualization for {modality}: {viz_error}")

            logger.info(f"  Successfully registered {len(atlas_result['registered_modalities'])} modalities to atlas")

        except Exception as e:
            logger.error(f"  [Error] Intra-study to atlas registration failed: {e}")
            # Continue anyway - atlas registration is optional
    elif atlas_method is not None and results["reference_modality"] is None:
        logger.warning("  [Sub-step 3b] Atlas registration skipped - no reference modality from step 3a")
    else:
        logger.info("  [Sub-step 3b] Intra-study to atlas registration skipped (method=None)")

    return results
