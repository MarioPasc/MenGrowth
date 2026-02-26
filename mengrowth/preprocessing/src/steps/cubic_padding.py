"""Cubic padding step: zero-pad volumes to cubic shape for registration stability.

This is a study-level step that:
1. Finds the maximum dimension across all modalities in the study
2. Pads each modality to a cube of that size with symmetric padding
3. Uses the minimum intensity value of each image as fill value

Scientific rationale:
- Normalizes field-of-view across different sequences (T1, T2, FLAIR)
- Reduces boundary artifacts during registration transforms
- Prevents edge clipping during affine rotations/translations
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
import logging
import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_output_dir,
    get_temp_path,
    get_visualization_path,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
) -> Dict[str, Any]:
    """Execute cubic padding step (study-level operation).

    This step pads all modalities in a study to a common cubic shape:
    1. First pass: Find max dimension across all modalities
    2. Second pass: Apply symmetric padding to each modality

    Args:
        context: Execution context (modality will be None for study-level steps)
        total_steps: Not used for study-level steps
        current_step_num: Not used for study-level steps

    Returns:
        Dict with padding results per modality:
        - original_shape: Original shape before padding
        - padded_shape: Shape after padding
        - target_size: Target cubic size used
        - padding_applied: Tuple of padding per axis [(before, after), ...]
        - fill_value: Fill value used for padding

    Raises:
        RuntimeError: If padding fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    patient_id = context.patient_id
    study_dir = context.study_dir

    # Check if cubic padding is enabled
    method = config.cubic_padding.method
    if method is None:
        logger.info(
            f"\n  Executing study-level step: {context.step_name} - skipped (method=None)"
        )
        return {}

    logger.info(f"\n  Executing study-level step: {context.step_name} ({method})")

    # Determine output directory based on mode
    study_output_dir = get_output_dir(context=context)
    viz_base = Path(orchestrator.config.viz_root) / patient_id / study_dir.name

    # =========================================================================
    # PASS 1: Find maximum dimension across all modalities
    # =========================================================================
    max_dimension = 0
    modality_shapes: Dict[str, Tuple[int, ...]] = {}
    modality_spacings: Dict[str, Tuple[float, ...]] = {}

    for modality in orchestrator.config.modalities:
        modality_path = study_output_dir / f"{modality}.nii.gz"

        if not modality_path.exists():
            logger.debug(f"  Skipping {modality} - file not found")
            continue

        img = nib.load(str(modality_path))
        shape = img.shape[:3]  # Handle 4D images
        spacing = img.header.get_zooms()[:3]

        modality_shapes[modality] = shape
        modality_spacings[modality] = spacing

        # Track maximum dimension
        current_max = max(shape)
        if current_max > max_dimension:
            max_dimension = current_max

        logger.debug(f"  {modality}: shape={shape}, max_dim={current_max}")

    if not modality_shapes:
        logger.warning("  No modalities found to pad")
        return {}

    logger.info(f"  Maximum dimension across modalities: {max_dimension}")

    # =========================================================================
    # PASS 2: Apply symmetric padding to each modality
    # =========================================================================
    results = {}

    for modality in orchestrator.config.modalities:
        modality_path = study_output_dir / f"{modality}.nii.gz"

        if modality not in modality_shapes:
            continue

        logger.info(f"  Processing {modality}...")

        original_shape = modality_shapes[modality]

        # Load image
        img = nib.load(str(modality_path))
        data = img.get_fdata()
        affine = img.affine.copy()

        # Handle 4D images (take first volume for processing)
        is_4d = len(data.shape) > 3
        if is_4d:
            data = data[..., 0]

        # Compute fill value based on configured mode
        if config.cubic_padding.fill_value_mode == "zero":
            fill_value = 0.0
        else:  # "min"
            fill_value = float(np.min(data))

        # Calculate symmetric padding for each axis
        padding = []
        for dim_size in original_shape:
            total_pad = max_dimension - dim_size
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.append((pad_before, pad_after))

        # Apply padding
        padded_data = np.pad(data, padding, mode="constant", constant_values=fill_value)

        # Update affine to account for the shift in origin
        # The origin shifts by -(pad_before * voxel_spacing) for each axis
        spacing = modality_spacings[modality]
        for i in range(3):
            # Shift origin in the negative direction by pad_before voxels
            affine[:3, 3] -= affine[:3, i] * padding[i][0]

        # Create new NIfTI image
        padded_img = nib.Nifti1Image(padded_data, affine, img.header)

        # Update header with new dimensions
        padded_img.header.set_data_shape(padded_data.shape)

        # Save to temporary file
        temp_padded = get_temp_path(context, modality, "cubic_padded")
        nib.save(padded_img, str(temp_padded))

        # Generate visualization if enabled
        if config.save_visualization:
            viz_output = get_visualization_path(context, suffix=f"_{modality}")
            _visualize_padding(
                original_path=modality_path,
                padded_path=temp_padded,
                output_path=viz_output,
                modality=modality,
                original_shape=original_shape,
                padded_shape=padded_data.shape,
                padding=padding,
                fill_value=fill_value,
            )

        # Replace original with padded version (in-place)
        temp_padded.replace(modality_path)

        # Store results
        results[modality] = {
            "original_shape": original_shape,
            "padded_shape": padded_data.shape,
            "target_size": max_dimension,
            "padding_applied": padding,
            "fill_value": fill_value,
        }

        logger.info(
            f"    {modality}: {original_shape} → {padded_data.shape} "
            f"(fill={fill_value:.2f})"
        )

    logger.info("  Cubic padding completed successfully")

    return results


def _visualize_padding(
    original_path: Path,
    padded_path: Path,
    output_path: Path,
    modality: str,
    original_shape: Tuple[int, ...],
    padded_shape: Tuple[int, ...],
    padding: List[Tuple[int, int]],
    fill_value: float,
) -> None:
    """Generate visualization comparing original and padded volumes.

    Creates a 3x2 grid:
    - Row 1: Axial view (original vs padded)
    - Row 2: Sagittal view (original vs padded)
    - Row 3: Coronal view (original vs padded)

    Shows voxel dimensions prominently on each panel.

    Args:
        original_path: Path to original image
        padded_path: Path to padded image
        output_path: Path to save visualization
        modality: Modality name
        original_shape: Original shape
        padded_shape: Padded shape
        padding: Padding applied per axis
        fill_value: Fill value used
    """
    try:
        # Load images
        orig_img = nib.load(str(original_path))
        padded_img = nib.load(str(padded_path))

        orig_data = orig_img.get_fdata()
        padded_data = padded_img.get_fdata()

        # Handle 4D
        if len(orig_data.shape) > 3:
            orig_data = orig_data[..., 0]
        if len(padded_data.shape) > 3:
            padded_data = padded_data[..., 0]

        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))

        # Compute display range (excluding fill value for better contrast)
        vmin = (
            np.percentile(orig_data[orig_data > fill_value + 1], 1)
            if np.any(orig_data > fill_value + 1)
            else np.min(orig_data)
        )
        vmax = (
            np.percentile(orig_data[orig_data > fill_value + 1], 99)
            if np.any(orig_data > fill_value + 1)
            else np.max(orig_data)
        )

        views = [
            ("Axial (Z)", 2, (0, 1)),  # Slice along z-axis, shows X×Y
            ("Sagittal (X)", 0, (1, 2)),  # Slice along x-axis, shows Y×Z
            ("Coronal (Y)", 1, (0, 2)),  # Slice along y-axis, shows X×Z
        ]

        for row, (view_name, axis, dims) in enumerate(views):
            # Get middle slices
            orig_slice_idx = orig_data.shape[axis] // 2
            padded_slice_idx = padded_data.shape[axis] // 2

            # Extract slices
            if axis == 0:
                orig_slice = orig_data[orig_slice_idx, :, :]
                padded_slice = padded_data[padded_slice_idx, :, :]
                orig_dims = (original_shape[1], original_shape[2])
                pad_dims = (padded_shape[1], padded_shape[2])
                dim_labels = ("Y", "Z")
            elif axis == 1:
                orig_slice = orig_data[:, orig_slice_idx, :]
                padded_slice = padded_data[:, padded_slice_idx, :]
                orig_dims = (original_shape[0], original_shape[2])
                pad_dims = (padded_shape[0], padded_shape[2])
                dim_labels = ("X", "Z")
            else:
                orig_slice = orig_data[:, :, orig_slice_idx]
                padded_slice = padded_data[:, :, padded_slice_idx]
                orig_dims = (original_shape[0], original_shape[1])
                pad_dims = (padded_shape[0], padded_shape[1])
                dim_labels = ("X", "Y")

            # Plot original
            axes[row, 0].imshow(
                orig_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax
            )
            axes[row, 0].set_title(
                f"{view_name}: Original\n"
                f"{dim_labels[0]}={orig_dims[0]} × {dim_labels[1]}={orig_dims[1]} voxels",
                fontsize=11,
                fontweight="bold",
            )
            axes[row, 0].set_xlabel(f"{dim_labels[0]} ({orig_dims[0]} vox)", fontsize=9)
            axes[row, 0].set_ylabel(f"{dim_labels[1]} ({orig_dims[1]} vox)", fontsize=9)
            axes[row, 0].tick_params(labelbottom=False, labelleft=False)

            # Plot padded with boundary rectangle
            axes[row, 1].imshow(
                padded_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax
            )
            axes[row, 1].set_title(
                f"{view_name}: Padded\n"
                f"{dim_labels[0]}={pad_dims[0]} × {dim_labels[1]}={pad_dims[1]} voxels",
                fontsize=11,
                fontweight="bold",
            )
            axes[row, 1].set_xlabel(f"{dim_labels[0]} ({pad_dims[0]} vox)", fontsize=9)
            axes[row, 1].set_ylabel(f"{dim_labels[1]} ({pad_dims[1]} vox)", fontsize=9)
            axes[row, 1].tick_params(labelbottom=False, labelleft=False)

            # Draw rectangle showing original content boundary within padded image
            if axis == 0:
                pad_x, pad_y = padding[1][0], padding[2][0]
                width, height = original_shape[1], original_shape[2]
            elif axis == 1:
                pad_x, pad_y = padding[0][0], padding[2][0]
                width, height = original_shape[0], original_shape[2]
            else:
                pad_x, pad_y = padding[0][0], padding[1][0]
                width, height = original_shape[0], original_shape[1]

            from matplotlib.patches import Rectangle

            rect = Rectangle(
                (pad_x - 0.5, pad_y - 0.5),
                width,
                height,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
                label="Original extent",
            )
            axes[row, 1].add_patch(rect)

        # Add main title with comprehensive info
        total_pad_voxels = sum(p[0] + p[1] for p in padding)
        fig.suptitle(
            f"Cubic Padding: {modality.upper()}\n"
            f"Original: {original_shape[0]}×{original_shape[1]}×{original_shape[2]} voxels  →  "
            f"Padded: {padded_shape[0]}×{padded_shape[1]}×{padded_shape[2]} voxels\n"
            f"Padding per axis: X=({padding[0][0]},{padding[0][1]}), "
            f"Y=({padding[1][0]},{padding[1][1]}), Z=({padding[2][0]},{padding[2][1]})  |  "
            f"Fill value: {fill_value:.2f}",
            fontsize=12,
            y=0.98,
        )

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="cyan",
                linestyle="--",
                linewidth=2,
                label="Original extent",
            )
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=10)

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"    Saved visualization: {output_path.name}")

    except Exception as e:
        logger.warning(f"    Failed to generate visualization: {e}")
