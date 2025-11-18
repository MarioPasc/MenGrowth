"""KDE-based normalization using intensity-normalization package.

This module implements kernel density estimation (KDE) based intensity normalization
that identifies the tissue mode via KDE and normalizes intensities relative to this mode.
This is particularly useful for multi-modal MRI where different tissue types have
characteristic intensity peaks.

Reference:
    intensity-normalization package: https://github.com/jcreinhold/intensity-normalization
"""

from pathlib import Path
from typing import Any, Dict
import logging

import nibabel as nib
import numpy as np
from scipy import stats

from intensity_normalization.normalize.kde import KDENormalize

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer
from mengrowth.preprocessing.src.normalization.utils import infer_modality_from_filename

logger = logging.getLogger(__name__)


class KDENormalizer(BaseNormalizer):
    """KDE-based tissue mode intensity normalizer using intensity-normalization package.

    This normalizer uses kernel density estimation to identify the dominant tissue
    mode (peak) in the intensity distribution and normalizes intensities relative
    to this mode.

    This uses the intensity-normalization package's normalize_image function with method="kde".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize KDE normalizer.

        Args:
            config: Configuration dictionary containing:
                - norm_value: Target value for the tissue mode (default=1.0)
            verbose: Enable verbose logging
        """
        super().__init__(
            config=config,
            verbose=verbose
        )

        # Extract norm_value parameter with default
        self.norm_value = config.get("norm_value", 1.0)

        if self.norm_value <= 0:
            raise ValueError(f"norm_value must be positive, got {self.norm_value}")

        self.logger.info(
            f"Initialized KDENormalizer: norm_value={self.norm_value}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute KDE-based normalization using intensity-normalization package.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - mask_path: Optional path to brain mask (Path)

        Returns:
            Dictionary containing:
                - 'mode': Tissue mode intensity value (KDE peak)
                - 'norm_value': Normalization target value
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
            # Import intensity-normalization package

            # Load NIfTI with nibabel
            self.logger.debug(f"Loading image: {input_path}")
            input_img = nib.load(str(input_path))
            input_data = input_img.get_fdata()

            # Store original range
            original_range = [float(input_data.min()), float(input_data.max())]

            # Apply KDE normalization using intensity-normalization package
            self.logger.info("Applying KDE normalization using intensity-normalization package...")
            normalizer = KDENormalize(norm_value=self.norm_value)
            modality = infer_modality_from_filename(input_path)
            self.logger.info(f"Inferred modality: {modality}, type: {type(modality)}, input: {input_path}")
            normalized_data = normalizer(input_data, modality=modality)


            # Store normalized range
            normalized_range = [float(normalized_data.min()), float(normalized_data.max())]

            self.logger.info(
                f"Original range: [{original_range[0]:.3f}, {original_range[1]:.3f}]"
            )
            self.logger.info(
                f"Normalized range: [{normalized_range[0]:.3f}, {normalized_range[1]:.3f}]"
            )

            # Save normalized image
            self.logger.debug(f"Saving normalized image: {output_path}")
            nib.Nifti1Image(normalized_data, input_img.affine).to_filename(str(output_path))


            self.logger.info("KDE normalization complete")

            return {
                "norm_value": self.norm_value,
                "original_range": original_range,
                "normalized_range": normalized_range,
            }

        except Exception as e:
            self.logger.error(f"KDE normalization failed: {e}")
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
        - Before and after intensity histograms with KDE overlay
        - Metadata text with mode value and ranges

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters containing metadata:
                - 'mode': Tissue mode intensity value
                - 'original_range': Original intensity range
                - 'normalized_range': Normalized intensity range

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mode_value = kwargs.get("mode")
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
                f'KDE-Based Normalization: {before_path.stem}',
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

            # Histogram for original with KDE overlay
            before_nonzero = before_data[before_data > 0]

            # Plot histogram
            axes[0, 3].hist(before_nonzero, bins=100, alpha=0.5, color='blue', density=True, label='Histogram')

            # Plot KDE if enough data points
            if len(before_nonzero) > 100:
                try:
                    # Subsample for KDE visualization if too many points
                    if len(before_nonzero) > 50000:
                        sample_idx = np.random.choice(len(before_nonzero), 50000, replace=False)
                        kde_data = before_nonzero[sample_idx]
                    else:
                        kde_data = before_nonzero

                    kde = stats.gaussian_kde(kde_data)
                    grid = np.linspace(before_nonzero.min(), before_nonzero.max(), 200)
                    axes[0, 3].plot(grid, kde(grid), 'r-', linewidth=2, label='KDE', alpha=0.7)
                except Exception:
                    pass  # Skip KDE plot if it fails

            axes[0, 3].axvline(mode_value, color='green', linestyle='--', linewidth=2, label=f'Mode={mode_value:.2f}')
            axes[0, 3].set_xlabel('Intensity', fontsize=10)
            axes[0, 3].set_ylabel('Density', fontsize=10)
            axes[0, 3].set_title('Original Histogram + KDE', fontsize=12)
            axes[0, 3].legend(fontsize=8)
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
            after_nonzero = after_data[after_data > vmin_after]
            axes[1, 3].hist(after_nonzero, bins=100, alpha=0.7, color='purple', density=True)
            axes[1, 3].axvline(self.norm_value, color='green', linestyle='--', linewidth=2, label=f'Target={self.norm_value:.2f}')
            axes[1, 3].set_xlabel('Intensity', fontsize=10)
            axes[1, 3].set_ylabel('Density', fontsize=10)
            axes[1, 3].set_title('Normalized Histogram', fontsize=12)
            axes[1, 3].legend(fontsize=8)
            axes[1, 3].grid(True, alpha=0.3)

            # Add metadata text
            metadata_text = (
                f"Normalization Method: KDE-Based\n"
                f"  Tissue Mode (KDE Peak) = {mode_value:.3f}\n"
                f"  Target Norm Value = {self.norm_value:.3f}\n\n"
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
