"""Percentile-based min-max normalization.

This module implements robust percentile-based intensity normalization that maps
intensity values to a comparable range across scans. This is a monotonic, robust
scaling method that preserves tissue ordering while reducing the influence of outliers.

Reference:
    Based on the normalization formula from the MenGrowth preprocessing README.
"""

from pathlib import Path
from typing import Any, Dict
import logging

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer

logger = logging.getLogger(__name__)


class PercentileMinMaxNormalizer(BaseNormalizer):
    """Percentile-based min-max intensity normalizer.

    This normalizer applies robust percentile-based scaling to map intensities
    to a comparable dynamic range. The transformation is:

        I'(x) = (I(x) - P_p1) / (P_p2 - P_p1)

    where P_p1 and P_p2 are the p1-th and p2-th percentiles of in-head intensities.
    This is monotonic, preserves tissue ordering, and reduces outlier influence.

    Typical values: p1=1.0, p2=99.0 (or p1=0.5, p2=99.5 for more conservatism).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize percentile min-max normalizer.

        Args:
            config: Configuration dictionary containing:
                - p1: Lower percentile threshold (default=1.0)
                - p2: Upper percentile threshold (default=99.0)
            verbose: Enable verbose logging
        """
        super().__init__(
            config=config,
            verbose=verbose
        )

        # Extract percentile parameters with defaults
        self.p1 = config.get("p1", 1.0)
        self.p2 = config.get("p2", 99.0)

        # Validate percentiles
        if not 0.0 <= self.p1 < self.p2 <= 100.0:
            raise ValueError(
                f"Percentiles must satisfy 0 <= p1 < p2 <= 100, got p1={self.p1}, p2={self.p2}"
            )

        self.logger.info(
            f"Initialized PercentileMinMaxNormalizer: p1={self.p1}, p2={self.p2}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute percentile-based min-max normalization.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - mask_path: Optional path to brain mask (Path)

        Returns:
            Dictionary containing:
                - 'p1_value': Lower percentile intensity value
                - 'p2_value': Upper percentile intensity value
                - 'p1_percentile': Lower percentile (p1)
                - 'p2_percentile': Upper percentile (p2)
                - 'original_range': Original intensity range [min, max]
                - 'normalized_range': Normalized intensity range [min, max]
                - 'num_voxels': Number of voxels processed

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If normalization fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)
        mask_path = kwargs.get("mask_path", None)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI with SimpleITK
            self.logger.debug(f"Loading image: {input_path}")
            image_sitk = sitk.ReadImage(str(input_path))
            image_array = sitk.GetArrayFromImage(image_sitk)

            # Compute percentiles
            p1_value = np.percentile(image_array, self.p1)
            p2_value = np.percentile(image_array, self.p2)

            self.logger.info(
                f"Percentiles: P{self.p1}={p1_value:.3f}, P{self.p2}={p2_value:.3f}"
            )

            # Avoid division by zero
            if p2_value == p1_value:
                self.logger.warning(
                    f"Percentile values are equal (P{self.p1}=P{self.p2}={p1_value:.3f}). "
                    "Normalization may not be meaningful."
                )
                p2_value = p1_value + 1e-8  # Add small epsilon

            # Apply normalization: I'(x) = (I(x) - p1_value) / (p2_value - p1_value)
            normalized_array = (image_array - p1_value) / (p2_value - p1_value)

            # Store original and normalized ranges
            original_range = [float(image_array.min()), float(image_array.max())]
            normalized_range = [float(normalized_array.min()), float(normalized_array.max())]

            self.logger.info(
                f"Original range: [{original_range[0]:.3f}, {original_range[1]:.3f}]"
            )
            self.logger.info(
                f"Normalized range: [{normalized_range[0]:.3f}, {normalized_range[1]:.3f}]"
            )

            # Create output image
            output_image = sitk.GetImageFromArray(normalized_array)
            output_image.CopyInformation(image_sitk)

            # Save normalized image
            self.logger.debug(f"Saving normalized image: {output_path}")
            sitk.WriteImage(output_image, str(output_path))

            self.logger.info("Percentile min-max normalization complete")

            return {
                "p1_value": float(p1_value),
                "p2_value": float(p2_value),
                "p1_percentile": self.p1,
                "p2_percentile": self.p2,
                "original_range": original_range,
                "normalized_range": normalized_range,
            }

        except Exception as e:
            self.logger.error(f"Percentile min-max normalization failed: {e}")
            raise RuntimeError(f"Normalization failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing before and after normalization.

        Creates visualization with:
        - Before and after image slices (axial, sagittal, coronal)
        - Before and after intensity histograms
        - Metadata text with percentile values and ranges

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters containing metadata:
                - 'p1_value': Lower percentile intensity value
                - 'p2_value': Upper percentile intensity value
                - 'original_range': Original intensity range
                - 'normalized_range': Normalized intensity range

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        p1_value = kwargs.get("p1_value")
        p2_value = kwargs.get("p2_value")
        original_range = kwargs.get("original_range")
        normalized_range = kwargs.get("normalized_range")

        self.logger.info(f"Generating normalization visualization: {output_path}")

        try:
            # Load images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Get middle slices for each view
            # Axial (XY plane, slice along Z)
            mid_z = before_data.shape[2] // 2
            axial_before = before_data[:, :, mid_z].T
            axial_after = after_data[:, :, mid_z].T

            # Sagittal (YZ plane, slice along X)
            mid_x = before_data.shape[0] // 2
            sagittal_before = before_data[mid_x, :, :].T
            sagittal_after = after_data[mid_x, :, :].T

            # Coronal (XZ plane, slice along Y)
            mid_y = before_data.shape[1] // 2
            coronal_before = before_data[:, mid_y, :].T
            coronal_after = after_data[:, mid_y, :].T

            # Create figure: 2 rows x 4 columns (3 views + 1 histogram per row)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(
                f'Percentile Min-Max Normalization: {before_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Row 1: Original image
            # Compute intensity range for original (for consistent display)
            vmin_before = before_data.min()
            vmax_before = before_data.max()

            axes[0, 0].imshow(axial_before, cmap='gray', origin='lower', vmin=vmin_before, vmax=vmax_before)
            axes[0, 0].set_title('Original - Axial', fontsize=12)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(sagittal_before, cmap='gray', origin='lower', vmin=vmin_before, vmax=vmax_before)
            axes[0, 1].set_title('Original - Sagittal', fontsize=12)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(coronal_before, cmap='gray', origin='lower', vmin=vmin_before, vmax=vmax_before)
            axes[0, 2].set_title('Original - Coronal', fontsize=12)
            axes[0, 2].axis('off')

            # Histogram for original
            before_nonzero = before_data[before_data > 0]
            axes[0, 3].hist(before_nonzero, bins=100, alpha=0.7, color='blue', density=True)
            axes[0, 3].axvline(p1_value, color='red', linestyle='--', linewidth=2, label=f'P{self.p1}')
            axes[0, 3].axvline(p2_value, color='green', linestyle='--', linewidth=2, label=f'P{self.p2}')
            axes[0, 3].set_xlabel('Intensity', fontsize=10)
            axes[0, 3].set_ylabel('Density', fontsize=10)
            axes[0, 3].set_title('Original Histogram', fontsize=12)
            axes[0, 3].legend(fontsize=9)
            axes[0, 3].grid(True, alpha=0.3)

            # Row 2: Normalized image
            # Compute intensity range for normalized
            vmin_after = after_data.min()
            vmax_after = after_data.max()

            axes[1, 0].imshow(axial_after, cmap='gray', origin='lower', vmin=vmin_after, vmax=vmax_after)
            axes[1, 0].set_title('Normalized - Axial', fontsize=12)
            axes[1, 0].axis('off')

            axes[1, 1].imshow(sagittal_after, cmap='gray', origin='lower', vmin=vmin_after, vmax=vmax_after)
            axes[1, 1].set_title('Normalized - Sagittal', fontsize=12)
            axes[1, 1].axis('off')

            axes[1, 2].imshow(coronal_after, cmap='gray', origin='lower', vmin=vmin_after, vmax=vmax_after)
            axes[1, 2].set_title('Normalized - Coronal', fontsize=12)
            axes[1, 2].axis('off')

            # Histogram for normalized
            after_nonzero = after_data[after_data > vmin_after]  # Exclude very negative values
            axes[1, 3].hist(after_nonzero, bins=100, alpha=0.7, color='orange', density=True)
            axes[1, 3].set_xlabel('Intensity', fontsize=10)
            axes[1, 3].set_ylabel('Density', fontsize=10)
            axes[1, 3].set_title('Normalized Histogram', fontsize=12)
            axes[1, 3].grid(True, alpha=0.3)

            # Add metadata text
            metadata_text = (
                f"Normalization Method: Percentile Min-Max\n"
                f"  P{self.p1} = {p1_value:.3f}\n"
                f"  P{self.p2} = {p2_value:.3f}\n\n"
                f"Original Range: [{original_range[0]:.3f}, {original_range[1]:.3f}]\n"
                f"Normalized Range: [{normalized_range[0]:.3f}, {normalized_range[1]:.3f}]"
            )

            fig.text(
                0.5, 0.01,
                metadata_text,
                ha='center',
                fontsize=10,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0.08, 1, 0.98])

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
