"""Data harmonization step: NRRD->NIfTI conversion, reorientation, intensity clipping, background removal.

This is a composite step that performs four sub-operations sequentially:
1. NRRD to NIfTI conversion
2. Reorientation to specified orientation (e.g., RAS)
3. Intensity outlier clipping (hot pixel removal)
4. Background removal (conditional)
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_temp_path,
    get_visualization_path,
    log_step_start,
)

logger = logging.getLogger(__name__)


def _visualize_intensity_clipping(
    nii_path: Path,
    threshold: Optional[float],
    n_clipped: int,
    output_path: Path,
    upper_percentile: float,
) -> None:
    """Generate histogram visualization showing intensity clipping effect.

    Args:
        nii_path: Path to the (already clipped) NIfTI file
        threshold: Clipping threshold value (None if no nonzero voxels)
        n_clipped: Number of voxels that were clipped
        output_path: Path to save the visualization PNG
        upper_percentile: Percentile used for clipping
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata()
    nonzero = data[data > 0]

    if len(nonzero) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(nonzero, bins=200, color="steelblue", alpha=0.7, log=True)
    if threshold is not None:
        ax.axvline(
            threshold,
            color="red",
            linestyle="--",
            label=f"P{upper_percentile} = {threshold:.1f}",
        )
        ax.legend()
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Intensity Distribution (clipped {n_clipped} voxels)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int,
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
    logger.info("        - Converting NRRD to NIfTI...")
    converter = orchestrator._get_component("converter", config)

    # Find input NRRD file
    input_file = context.study_dir / f"{modality}.nrrd"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    converter.execute(
        input_file,
        paths["nifti"],
        allow_overwrite=orchestrator.config.overwrite,
    )

    if config.save_visualization:
        viz_path = get_visualization_path(context, suffix="_convert")
        converter.visualize(input_file, paths["nifti"], viz_path)

    # Sub-step 1.5: Detect and fix transposed/corrupted axes
    # Some NRRD files have the slice dimension misidentified (e.g., 512x28x512
    # instead of 512x512x28). Detect by checking if one spatial dimension has
    # vastly fewer voxels than the others and is NOT in the z-position.
    conv_nii = nib.load(str(paths["nifti"]))
    conv_shape = conv_nii.shape[:3]
    min_dim = min(conv_shape)
    max_dim = max(conv_shape)
    min_axis = conv_shape.index(min_dim)

    if min_dim < 40 and max_dim / min_dim > 10 and min_axis != 2:
        logger.warning(
            f"        - Detected likely transposed axes: shape={conv_shape}, "
            f"axis {min_axis} has only {min_dim} voxels. "
            f"Transposing to move thin axis to z-position."
        )
        conv_data = conv_nii.get_fdata()
        conv_data = np.moveaxis(conv_data, min_axis, 2)
        # Swap the corresponding affine columns to keep spatial mapping correct
        new_affine = conv_nii.affine.copy()
        new_affine[:, [min_axis, 2]] = new_affine[:, [2, min_axis]]
        fixed_nii = nib.Nifti1Image(conv_data, new_affine, conv_nii.header)
        temp_transposed = get_temp_path(context, modality, "transposed")
        nib.save(fixed_nii, str(temp_transposed))
        temp_transposed.replace(paths["nifti"])
        logger.info(f"          Fixed: new shape={conv_data.shape}")

    # Sub-step 2: Reorientation
    logger.info(f"        - Reorienting to {config.reorient_to}...")
    reorienter = orchestrator._get_component("reorienter", config)
    temp_reoriented = get_temp_path(context, modality, "reoriented")

    reorienter.execute(paths["nifti"], temp_reoriented, allow_overwrite=True)

    if config.save_visualization:
        viz_path = get_visualization_path(context, suffix="_reorient")
        reorienter.visualize(paths["nifti"], temp_reoriented, viz_path)

    # Replace original with reoriented
    temp_reoriented.replace(paths["nifti"])

    # Sub-step 3: Intensity outlier clipping (hot pixel removal)
    clip_config = config.intensity_clipping
    if clip_config.enabled:
        logger.info(
            f"        - Clipping intensity outliers (P{clip_config.upper_percentile})..."
        )

        nii = nib.load(str(paths["nifti"]))
        data = nii.get_fdata().copy()

        nonzero_mask = data > 0
        n_clipped = 0
        threshold = None

        if nonzero_mask.any():
            threshold = float(
                np.percentile(data[nonzero_mask], clip_config.upper_percentile)
            )
            clipped_mask = data > threshold
            n_clipped = int(clipped_mask.sum())

            if n_clipped > 0:
                data[clipped_mask] = threshold

                if clip_config.log_clipped_count:
                    n_nonzero = int(nonzero_mask.sum())
                    logger.info(
                        f"          Clipped {n_clipped} voxels "
                        f"({n_clipped / n_nonzero:.4%} of nonzero) "
                        f"above threshold {threshold:.2f}"
                    )

                # Write via temp file
                temp_clipped = get_temp_path(context, modality, "clipped")
                clipped_nii = nib.Nifti1Image(data, nii.affine, nii.header)
                nib.save(clipped_nii, str(temp_clipped))
                temp_clipped.replace(paths["nifti"])
            else:
                logger.info(
                    f"          No voxels above P{clip_config.upper_percentile} threshold"
                )
        else:
            logger.info("          No nonzero voxels — skipping clipping")

        if config.save_visualization:
            viz_path = get_visualization_path(context, suffix="_intensity_clip")
            _visualize_intensity_clipping(
                nii_path=paths["nifti"],
                threshold=threshold,
                n_clipped=n_clipped,
                output_path=viz_path,
                upper_percentile=clip_config.upper_percentile,
            )
    else:
        logger.info("        - Intensity clipping skipped (disabled)")

    # Sub-step 4: Background removal (if enabled)
    bg_method = config.background_zeroing.method
    if bg_method is not None:
        logger.info(f"        - Removing background ({bg_method})...")
        bg_remover = orchestrator._get_component(
            f"background_remover_{bg_method}", config
        )
        temp_masked = get_temp_path(context, modality, "masked")

        bg_remover.execute(paths["nifti"], temp_masked, allow_overwrite=True)

        if config.save_visualization:
            viz_path = get_visualization_path(context, suffix="_background")
            bg_remover.visualize(paths["nifti"], temp_masked, viz_path)

        # Replace with masked version
        temp_masked.replace(paths["nifti"])
    else:
        logger.info("        - Background removal skipped (method=None)")

    return results
