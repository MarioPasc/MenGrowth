"""Reorientation operations for data harmonization.

This module implements reorientation converters that wrap existing utilities
in mengrowth.preprocessing.src.utils following the OOP design pattern.
"""

from pathlib import Path
from typing import Any, Literal
import logging
import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.step0_data_harmonization.base import BaseReorienter

logger = logging.getLogger(__name__)


class Reorienter(BaseReorienter):
    """Reorient NIfTI volumes to target orientation (RAS or LPS).

    This reorienter wraps SimpleITK's DICOMOrientImageFilter to transform
    volumes to a target coordinate system while preserving physical spacing
    and affine transforms.
    """

    def __init__(
        self,
        target_orientation: Literal["RAS", "LPS"],
        verbose: bool = False
    ) -> None:
        """Initialize reorienter.

        Args:
            target_orientation: Target orientation ("RAS" or "LPS")
            verbose: Enable verbose logging

        Raises:
            ValueError: If target_orientation is not "RAS" or "LPS"
        """
        super().__init__(target_orientation=target_orientation, verbose=verbose)
        self.logger.info(f"Initialized Reorienter with target: {target_orientation}")

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Reorient NIfTI volume to target orientation.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output NIfTI file
            **kwargs: Additional parameters (allow_overwrite)

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If reorientation fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Read image with SimpleITK
            img = sitk.ReadImage(str(input_path))

            # Get current orientation for logging
            current_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                img.GetDirection()
            )

            self.logger.info(
                f"Reorienting from {current_orientation} to {self.target_orientation}"
            )

            # Create reorientation filter
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(self.target_orientation)

            # Apply reorientation
            reoriented_img = orient_filter.Execute(img)

            # Write output
            sitk.WriteImage(reoriented_img, str(output_path))

            self.logger.info(
                f"Successfully reoriented {input_path.name} to {self.target_orientation}"
            )

        except Exception as e:
            self.logger.error(f"Reorientation failed: {e}")
            raise RuntimeError(f"Reorientation failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing before and after reorientation.

        Displays axial, sagittal, and coronal mid-slices from both
        orientations to verify the transformation.

        Args:
            before_path: Path to input file (before reorientation)
            after_path: Path to output file (after reorientation)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional visualization parameters

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.logger.info(f"Generating reorientation visualization: {output_path}")

        try:
            # Load before and after
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Get orientations for labels
            before_orient = nib.aff2axcodes(before_img.affine)
            after_orient = nib.aff2axcodes(after_img.affine)

            # Get mid-slices
            before_mid_x = before_data.shape[0] // 2
            before_mid_y = before_data.shape[1] // 2
            before_mid_z = before_data.shape[2] // 2

            after_mid_x = after_data.shape[0] // 2
            after_mid_y = after_data.shape[1] // 2
            after_mid_z = after_data.shape[2] // 2

            # Create figure: 2 rows x 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f'Reorientation: {before_path.stem}\n'
                f'{"".join(before_orient)} → {"".join(after_orient)}',
                fontsize=16,
                fontweight='bold'
            )

            # Before reorientation (row 0)
            axes[0, 0].imshow(before_data[:, :, before_mid_z].T, cmap='gray', origin='lower')
            axes[0, 0].set_title(f'Before - Axial ({"".join(before_orient)})')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(before_data[:, before_mid_y, :].T, cmap='gray', origin='lower')
            axes[0, 1].set_title(f'Before - Sagittal')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(before_data[before_mid_x, :, :].T, cmap='gray', origin='lower')
            axes[0, 2].set_title(f'Before - Coronal')
            axes[0, 2].axis('off')

            # After reorientation (row 1)
            axes[1, 0].imshow(after_data[:, :, after_mid_z].T, cmap='gray', origin='lower')
            axes[1, 0].set_title(f'After - Axial ({self.target_orientation})')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(after_data[:, after_mid_y, :].T, cmap='gray', origin='lower')
            axes[1, 1].set_title(f'After - Sagittal')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(after_data[after_mid_x, :, :].T, cmap='gray', origin='lower')
            axes[1, 2].set_title(f'After - Coronal')
            axes[1, 2].axis('off')

            plt.tight_layout()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
