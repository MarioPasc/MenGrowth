"""BSpline resampling using SimpleITK.

This module implements BSpline interpolation-based resampling following the OOP
design pattern specified in CLAUDE.md. It wraps SimpleITK's ResampleImageFilter
with BSpline interpolation for high-quality image resampling.
"""

from pathlib import Path
from typing import Any, Dict, List
import logging

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.resampling.base import BaseResampler

logger = logging.getLogger(__name__)


class BSplineResampler(BaseResampler):
    """BSpline interpolation-based resampler using SimpleITK.

    This resampler uses BSpline interpolation to resample volumes to a target
    voxel spacing while preserving image quality and smooth intensity transitions.
    BSpline interpolation is particularly well-suited for medical imaging as it
    provides smooth derivatives and reduces aliasing artifacts.

    Reference:
        Unser M, et al. "B-spline signal processing" IEEE TSP 1993.
    """

    def __init__(
        self,
        target_voxel_size: List[float],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize BSpline resampler.

        Args:
            target_voxel_size: Target voxel size in mm [x, y, z]
            config: Configuration dictionary containing:
                - bspline_order: BSpline order [0-5] (default=3)
                  0: nearest neighbor
                  1: linear
                  3: cubic (recommended for MRI)
                  5: quintic (higher quality, slower)
            verbose: Enable verbose logging
        """
        super().__init__(
            target_voxel_size=target_voxel_size,
            config=config,
            verbose=verbose
        )

        # Extract BSpline order with default
        self.bspline_order = config.get("bspline_order", 3)

        # Validate order
        if not isinstance(self.bspline_order, int) or not (0 <= self.bspline_order <= 5):
            raise ValueError(
                f"bspline_order must be an integer in [0, 5], got {self.bspline_order}"
            )

        self.logger.info(
            f"Initialized BSplineResampler: order={self.bspline_order}, "
            f"target_spacing={target_voxel_size}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute BSpline resampling.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output resampled NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)

        Returns:
            Dictionary containing:
                - 'original_spacing': Original voxel spacing [x, y, z]
                - 'target_spacing': Target voxel spacing [x, y, z]
                - 'original_shape': Original image shape [x, y, z]
                - 'resampled_shape': Resampled image shape [x, y, z]

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If resampling fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI with SimpleITK
            self.logger.debug(f"Loading image: {input_path}")
            image_sitk = sitk.ReadImage(str(input_path))

            # Get original spacing and size
            original_spacing = np.array(image_sitk.GetSpacing())
            original_size = np.array(image_sitk.GetSize())

            self.logger.info(
                f"Original spacing: {original_spacing}, shape: {original_size}"
            )

            # Compute new size based on target spacing
            # new_size = original_size * (original_spacing / target_spacing)
            target_spacing = np.array(self.target_voxel_size)
            new_size = np.round(original_size * (original_spacing / target_spacing)).astype(int)

            self.logger.info(
                f"Target spacing: {target_spacing}, new shape: {new_size}"
            )

            # Set up resampler
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing.tolist())
            resampler.SetSize(new_size.tolist())
            resampler.SetOutputDirection(image_sitk.GetDirection())
            resampler.SetOutputOrigin(image_sitk.GetOrigin())
            resampler.SetTransform(sitk.Transform())  # Identity transform
            resampler.SetDefaultPixelValue(0.0)

            # Set BSpline interpolator
            resampler.SetInterpolator(sitk.sitkBSpline)

            # Note: SimpleITK's BSpline interpolator uses order 3 by default
            # For different orders, we would need to use sitkNearestNeighbor (0),
            # sitkLinear (1), or sitkBSpline (3)
            if self.bspline_order == 0:
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                self.logger.debug("Using Nearest Neighbor interpolation (order 0)")
            elif self.bspline_order == 1:
                resampler.SetInterpolator(sitk.sitkLinear)
                self.logger.debug("Using Linear interpolation (order 1)")
            elif self.bspline_order == 3:
                resampler.SetInterpolator(sitk.sitkBSpline)
                self.logger.debug("Using BSpline interpolation (order 3)")
            else:
                # For orders 2, 4, 5, default to BSpline (order 3) with a warning
                self.logger.warning(
                    f"BSpline order {self.bspline_order} not directly supported by SimpleITK. "
                    "Using BSpline order 3 instead."
                )
                resampler.SetInterpolator(sitk.sitkBSpline)

            # Execute resampling
            self.logger.info(f"Resampling with BSpline order {self.bspline_order}...")
            resampled_image = resampler.Execute(image_sitk)

            # Save resampled image
            self.logger.debug(f"Saving resampled image: {output_path}")
            sitk.WriteImage(resampled_image, str(output_path))

            # Verify output spacing
            output_spacing = np.array(resampled_image.GetSpacing())
            output_size = np.array(resampled_image.GetSize())

            self.logger.info(
                f"Resampling complete: output spacing={output_spacing}, "
                f"output shape={output_size}"
            )

            return {
                "original_spacing": original_spacing.tolist(),
                "target_spacing": target_spacing.tolist(),
                "original_shape": original_size.tolist(),
                "resampled_shape": output_size.tolist()
            }

        except Exception as e:
            self.logger.error(f"BSpline resampling failed: {e}")
            raise RuntimeError(f"Resampling failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate 3-view (axial, sagittal, coronal) comparison visualization.

        Creates visualization with:
        - Row 1: Original image (axial, sagittal, coronal)
        - Row 2: Resampled image (axial, sagittal, coronal)
        - Metadata text with spacing and shape information

        Args:
            before_path: Path to input file (before resampling)
            after_path: Path to output file (after resampling)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters:
                - 'original_spacing': Original voxel spacing
                - 'target_spacing': Target voxel spacing
                - 'original_shape': Original image shape
                - 'resampled_shape': Resampled image shape

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        original_spacing = kwargs.get("original_spacing")
        target_spacing = kwargs.get("target_spacing")
        original_shape = kwargs.get("original_shape")
        resampled_shape = kwargs.get("resampled_shape")

        self.logger.info(f"Generating resampling visualization: {output_path}")

        try:
            # Load images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Get middle slices for each view
            # Axial (XY plane, slice along Z)
            mid_z_before = before_data.shape[2] // 2
            mid_z_after = after_data.shape[2] // 2
            axial_before = before_data[:, :, mid_z_before].T
            axial_after = after_data[:, :, mid_z_after].T

            # Sagittal (YZ plane, slice along X)
            mid_x_before = before_data.shape[0] // 2
            mid_x_after = after_data.shape[0] // 2
            sagittal_before = before_data[mid_x_before, :, :].T
            sagittal_after = after_data[mid_x_after, :, :].T

            # Coronal (XZ plane, slice along Y)
            mid_y_before = before_data.shape[1] // 2
            mid_y_after = after_data.shape[1] // 2
            coronal_before = before_data[:, mid_y_before, :].T
            coronal_after = after_data[:, mid_y_after, :].T

            # Create figure: 2 rows x 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(
                f'Resampling (BSpline order {self.bspline_order}): {before_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Compute shared intensity range for consistent visualization
            vmin = min(before_data.min(), after_data.min())
            vmax = max(before_data.max(), after_data.max())

            # Row 1: Original image
            axes[0, 0].imshow(axial_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title('Original - Axial')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(sagittal_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title('Original - Sagittal')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(coronal_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[0, 2].set_title('Original - Coronal')
            axes[0, 2].axis('off')

            # Row 2: Resampled image
            axes[1, 0].imshow(axial_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 0].set_title('Resampled - Axial')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(sagittal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 1].set_title('Resampled - Sagittal')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(coronal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 2].set_title('Resampled - Coronal')
            axes[1, 2].axis('off')

            # Add metadata text
            metadata_text = (
                f"Original:\n"
                f"  Spacing: {original_spacing}\n"
                f"  Shape: {original_shape}\n\n"
                f"Resampled:\n"
                f"  Spacing: {target_spacing}\n"
                f"  Shape: {resampled_shape}"
            )

            fig.text(
                0.5, 0.02,
                metadata_text,
                ha='center',
                fontsize=10,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0.05, 1, 0.98])

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
