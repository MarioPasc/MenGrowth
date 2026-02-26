"""Registration step: intra-study coregistration and atlas registration.

This is a study-level step that operates on all modalities together:
1. Intra-study multi-modal coregistration to reference modality
2. Intra-study to atlas registration (optional)
3. Propagate brain masks to atlas space (if skull stripping ran before registration)
"""

import shutil
from typing import Dict, Any, List
from pathlib import Path
import logging


from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_output_dir,
)

logger = logging.getLogger(__name__)


def _propagate_brain_masks_to_atlas_space(
    atlas_path: Path,
    transform_path: str,
    artifacts_dir: Path,
    modalities: List[str],
) -> int:
    """Propagate skull-stripping brain masks to atlas space after registration.

    When skull stripping runs BEFORE registration (v2 pipeline), brain masks
    are created in subject space. After atlas registration transforms modality
    images to atlas space, these masks must also be transformed. Using the
    actual registration transform (Rigid+Affine) is critical — a naive
    scipy.ndimage.zoom would only scale without applying rotation, causing
    systematic mask misalignment.

    NearestNeighbor interpolation preserves binary mask integrity (no partial
    volume artifacts at mask boundaries).

    Args:
        atlas_path: Path to atlas NIfTI file (defines output grid).
        transform_path: Path to ref→atlas composite transform (.h5).
        artifacts_dir: Directory containing brain mask artifacts.
        modalities: List of modality names to search for masks.

    Returns:
        Number of masks successfully propagated.
    """
    try:
        import ants
    except ImportError:
        logger.warning(
            "  ANTsPy not available — cannot propagate brain masks to atlas space"
        )
        return 0

    # Collect all brain mask files in artifacts
    mask_files: set[Path] = set()
    for modality in modalities:
        for suffix in ("_brain_mask.nii.gz", "_brain_mask_individual.nii.gz"):
            candidate = artifacts_dir / f"{modality}{suffix}"
            if candidate.exists():
                mask_files.add(candidate)

    consensus = artifacts_dir / "consensus_brain_mask.nii.gz"
    if consensus.exists():
        mask_files.add(consensus)

    if not mask_files:
        logger.debug("  No brain masks found in artifacts — skipping propagation")
        return 0

    logger.info(f"  Propagating {len(mask_files)} brain mask(s) to atlas space")

    atlas_img = ants.image_read(str(atlas_path))
    propagated = 0

    for mask_file in sorted(mask_files):
        try:
            mask_img = ants.image_read(str(mask_file))
            transformed = ants.apply_transforms(
                fixed=atlas_img,
                moving=mask_img,
                transformlist=[str(transform_path)],
                interpolator="nearestNeighbor",
                defaultvalue=0,
            )

            # Atomic write via temp file
            temp_path = mask_file.parent / f"_temp_{mask_file.name}"
            ants.image_write(transformed, str(temp_path))
            temp_path.replace(mask_file)

            logger.info(f"    Propagated: {mask_file.name}")
            propagated += 1
        except Exception as e:
            logger.warning(f"    Failed to propagate {mask_file.name}: {e}")

    return propagated


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
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
        "registered_modalities": [],
    }

    # Determine study output directory based on mode
    study_output_dir = get_output_dir(context=context)

    # ── Diagnostic: log file state BEFORE registration reads anything ──
    import nibabel as nib
    import numpy as np

    for mod in orchestrator.config.modalities:
        mod_path = study_output_dir / f"{mod}.nii.gz"
        if mod_path.exists():
            diag_nii = nib.load(str(mod_path))
            diag_data = diag_nii.get_fdata()
            nonzero_frac = np.count_nonzero(diag_data) / diag_data.size
            logger.info(
                f"  [DIAG] {mod} BEFORE registration: "
                f"shape={diag_data.shape}, "
                f"nonzero={nonzero_frac:.3%}, "
                f"range=[{diag_data.min():.2f}, {diag_data.max():.2f}], "
                f"path={mod_path}"
            )
    artifacts_base = (
        Path(orchestrator.config.preprocessing_artifacts_path)
        / patient_id
        / study_dir.name
    )
    viz_base = Path(orchestrator.config.viz_root) / patient_id / study_dir.name

    # Step 3a: Intra-study multi-modal coregistration to reference
    intra_study_transforms = {}
    intra_study_method = (
        config.intra_study_to_reference.method
        if hasattr(config, "intra_study_to_reference")
        else None
    )

    if intra_study_method is not None:
        logger.info(
            f"  [Sub-step 3a] Intra-study multi-modal coregistration ({intra_study_method})"
        )

        # Get or create intra-study registrator
        intra_study_registrator = orchestrator._get_component(
            f"intra_study_to_ref_{context.step_name}", config
        )

        # Save pre-registration copies for visualization (execute() replaces in-place)
        pre_reg_copies: Dict[str, Path] = {}
        if config.save_visualization:
            for modality in orchestrator.config.modalities:
                modality_path = study_output_dir / f"{modality}.nii.gz"
                if modality_path.exists():
                    pre_reg_path = (
                        study_output_dir / f"_temp_{modality}_pre_registration.nii.gz"
                    )
                    shutil.copy2(str(modality_path), str(pre_reg_path))
                    pre_reg_copies[modality] = pre_reg_path

        try:
            # Execute intra-study to reference registration
            reg_result = intra_study_registrator.execute(
                study_dir=study_output_dir,
                artifacts_dir=artifacts_base,
                modalities=orchestrator.config.modalities,
            )

            # Store results
            results["reference_modality"] = reg_result["reference_modality"]
            results["transforms"] = reg_result["transforms"]
            results["registered_modalities"] = reg_result.get(
                "registered_modalities", []
            )
            intra_study_transforms = reg_result["transforms"]

            # Update orchestrator state
            orchestrator.selected_reference_modality = reg_result["reference_modality"]

            logger.info(f"  Reference modality: {results['reference_modality']}")

            # Generate visualizations if enabled
            if config.save_visualization:
                reference_path = (
                    study_output_dir / f"{results['reference_modality']}.nii.gz"
                )

                for modality in reg_result["registered_modalities"]:
                    registered_path = study_output_dir / f"{modality}.nii.gz"
                    moving_path = pre_reg_copies.get(modality, registered_path)
                    transform_path = reg_result["transforms"].get(modality)

                    viz_output = (
                        viz_base
                        / f"{context.step_name}_3a_intra_study_{modality}_to_{results['reference_modality']}.png"
                    )

                    try:
                        intra_study_registrator.visualize(
                            reference_path=reference_path,
                            moving_path=moving_path,
                            registered_path=registered_path,
                            output_path=viz_output,
                            modality=modality,
                            transform_path=transform_path,
                        )
                    except Exception as viz_error:
                        logger.warning(
                            f"  Failed to generate visualization for {modality}: {viz_error}"
                        )

            logger.info(
                f"  Successfully registered {len(reg_result['registered_modalities'])} modalities to reference"
            )

        except Exception as e:
            logger.error(f"  [Error] Intra-study to reference registration failed: {e}")
            raise RuntimeError("Intra-study registration failed") from e
        finally:
            # Clean up pre-registration temp copies
            for temp_path in pre_reg_copies.values():
                if temp_path.exists():
                    temp_path.unlink()
    else:
        logger.info(
            "  [Sub-step 3a] Intra-study to reference registration skipped (method=None)"
        )

    # Step 3b: Intra-study to atlas registration
    atlas_method = (
        config.intra_study_to_atlas.method
        if hasattr(config, "intra_study_to_atlas")
        else None
    )

    if atlas_method is not None and results["reference_modality"] is not None:
        logger.info(
            f"  [Sub-step 3b] Intra-study to atlas registration ({atlas_method})"
        )

        # Get or create atlas registrator
        atlas_registrator = orchestrator._get_component(
            f"intra_study_to_atlas_{context.step_name}", config
        )

        try:
            # Execute atlas registration
            atlas_result = atlas_registrator.execute(
                study_dir=study_output_dir,
                artifacts_dir=artifacts_base,
                modalities=orchestrator.config.modalities,
                intra_study_transforms=intra_study_transforms,
            )

            logger.info(
                f"  Reference registered to atlas: {atlas_result['atlas_path']}"
            )

            # Generate visualizations if enabled
            if config.save_visualization:
                atlas_path = Path(config.intra_study_to_atlas.atlas_path)
                reference_path = (
                    study_output_dir / f"{results['reference_modality']}.nii.gz"
                )

                # Visualize reference to atlas
                viz_output_ref = (
                    viz_base
                    / f"{context.step_name}_3b_atlas_ref_{results['reference_modality']}.png"
                )
                try:
                    atlas_registrator.visualize(
                        atlas_path=atlas_path,
                        reference_path=reference_path,
                        output_path=viz_output_ref,
                        ref_to_atlas_transform=atlas_result["ref_to_atlas_transform"],
                    )
                except Exception as viz_error:
                    logger.warning(
                        f"  Failed to generate reference→atlas visualization: {viz_error}"
                    )

                # Visualize each modality in atlas space
                for modality in atlas_result["registered_modalities"]:
                    modality_path = study_output_dir / f"{modality}.nii.gz"
                    viz_output = (
                        viz_base / f"{context.step_name}_3b_atlas_{modality}.png"
                    )

                    try:
                        atlas_registrator.visualize(
                            atlas_path=atlas_path,
                            modality_path=modality_path,
                            output_path=viz_output,
                            modality=modality,
                        )
                    except Exception as viz_error:
                        logger.warning(
                            f"  Failed to generate atlas visualization for {modality}: {viz_error}"
                        )

            logger.info(
                f"  Successfully registered {len(atlas_result['registered_modalities'])} modalities to atlas"
            )

            # Add quality metrics from atlas registration if available
            if "quality_metrics" in atlas_result:
                results["quality_metrics"] = atlas_result["quality_metrics"]

            # Propagate brain masks to atlas space (critical for skull-strip-first pipelines)
            # If skull stripping ran before registration, masks are in subject space but
            # images are now in atlas space. Downstream steps (intensity normalization)
            # need masks in atlas space for correct brain voxel identification.
            ref_to_atlas_transform = atlas_result.get("ref_to_atlas_transform")
            if ref_to_atlas_transform:
                _propagate_brain_masks_to_atlas_space(
                    atlas_path=Path(config.intra_study_to_atlas.atlas_path),
                    transform_path=ref_to_atlas_transform,
                    artifacts_dir=artifacts_base,
                    modalities=orchestrator.config.modalities,
                )

            # ── Diagnostic: log file state AFTER atlas registration ──
            for mod in orchestrator.config.modalities:
                mod_path = study_output_dir / f"{mod}.nii.gz"
                if mod_path.exists():
                    diag_nii = nib.load(str(mod_path))
                    diag_data = diag_nii.get_fdata()
                    nonzero_frac = np.count_nonzero(diag_data) / diag_data.size
                    logger.info(
                        f"  [DIAG] {mod} AFTER atlas registration: "
                        f"shape={diag_data.shape}, "
                        f"nonzero={nonzero_frac:.3%}, "
                        f"range=[{diag_data.min():.2f}, {diag_data.max():.2f}]"
                    )

        except Exception as e:
            logger.error(f"  [Error] Intra-study to atlas registration failed: {e}")
            # Continue anyway - atlas registration is optional
    elif atlas_method is not None and results["reference_modality"] is None:
        logger.warning(
            "  [Sub-step 3b] Atlas registration skipped - no reference modality from step 3a"
        )
    else:
        logger.info(
            "  [Sub-step 3b] Intra-study to atlas registration skipped (method=None)"
        )

    # Add qc_paths for QC system (study-level step output paths)
    results["qc_paths"] = {
        "study_output_dir": str(study_output_dir),
        "reference_modality": results["reference_modality"],
        "atlas_path": str(config.intra_study_to_atlas.atlas_path)
        if hasattr(config, "intra_study_to_atlas")
        and config.intra_study_to_atlas.atlas_path
        else None,
        "modality_outputs": {
            mod: str(study_output_dir / f"{mod}.nii.gz")
            for mod in results.get("registered_modalities", [])
        },
    }

    return results
