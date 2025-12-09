"""LSQ-based population normalization using intensity-normalization package.

This module implements Least-Squares (LSQ) population-based intensity normalization
which fits tissue means across a population of images and normalizes individual images
based on the population standard. This is particularly useful for standardizing
intensities across cohorts of images.

Reference:
    intensity-normalization package: https://github.com/jcreinhold/intensity-normalization
    LSQ paper: "Least-squares tissue mean normalization for MRI"
"""

from pathlib import Path
from typing import Any, Dict, List
import logging

import nibabel as nib
import numpy as np
from scipy import stats

from intensity_normalization.normalizers.population.lsq import LSQNormalizer as LSQNormalize

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer
from mengrowth.preprocessing.src.normalization.utils import infer_modality_from_filename

logger = logging.getLogger(__name__)


class LSQNormalizer(BaseNormalizer):
    """Least-squares population-based intensity normalizer.

    This normalizer is POPULATION-BASED, meaning it must be fitted on a collection
    of images before it can normalize individual images. The fitting process establishes
    standard tissue means from the population, which are then used to normalize each image.

    This uses the intensity-normalization package's LSQNormalizer.

    Important:
        - Must call fit_population() before execute()
        - The fitted normalizer is stored and reused for all images in the population
    """

    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize LSQ normalizer.

        Args:
            config: Configuration dictionary containing:
                - norm_value: Scaling factor after normalization (default=1.0)
            verbose: Enable verbose logging
        """
        super().__init__(
            config=config,
            verbose=verbose
        )

        # Extract parameters with defaults
        self.norm_value = config.get("norm_value", 1.0)

        # Validate parameters
        if self.norm_value <= 0:
            raise ValueError(f"norm_value must be positive, got {self.norm_value}")

        # Initialize fitted normalizer as None (must be fitted before use)
        self.fitted_normalizer = None

        self.logger.info(
            f"Initialized LSQNormalizer: norm_value={self.norm_value}"
        )

    def fit_population(
        self,
        image_paths: List[Path]
    ) -> Dict[str, Any]:
        """Fit the LSQ normalizer on a population of images.

        This method must be called before execute(). It loads all images,
        fits the LSQ normalizer to establish standard tissue means, and
        stores the fitted normalizer for use in execute().

        Args:
            image_paths: List of paths to NIfTI files to fit on

        Returns:
            Dictionary containing:
                - 'num_images_fitted': Number of images used for fitting
                - 'norm_value': Normalization value used
                - 'image_paths': List of paths fitted on (as strings)

        Raises:
            ValueError: If image_paths is empty
            RuntimeError: If fitting fails
        """
        if len(image_paths) == 0:
            raise ValueError("Cannot fit LSQ normalizer on empty image list")

        self.logger.info(
            f"Fitting LSQ normalizer on population of {len(image_paths)} images..."
        )

        try:
            # Load all images
            images = []
            for img_path in image_paths:
                self.logger.debug(f"Loading image for fitting: {img_path}")
                img = nib.load(str(img_path))
                img_data = img.get_fdata()
                images.append(img_data)

            # Create LSQ normalizer
            normalizer = LSQNormalize(norm_value=self.norm_value)

            # Fit on population
            self.logger.info("Fitting LSQ normalizer on loaded images...")
            # The LSQNormalizer.fit() method expects a list of images
            normalizer.fit(images)

            # Store fitted normalizer
            self.fitted_normalizer = normalizer

            self.logger.info(
                f"LSQ normalizer fitted successfully on {len(images)} images"
            )

            return {
                "num_images_fitted": len(images),
                "norm_value": self.norm_value,
                "image_paths": [str(p) for p in image_paths],
            }

        except Exception as e:
            self.logger.error(f"LSQ population fitting failed: {e}")
            raise RuntimeError(f"Population fitting failed: {e}") from e

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute LSQ normalization using fitted population parameters.

        Important: fit_population() must be called before this method.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - mask_path: Optional path to brain mask (Path)

        Returns:
            Dictionary containing:
                - 'norm_value': Normalization value used
                - 'fitted': Whether normalizer was fitted (always True here)
                - 'modality': Inferred modality
                - 'original_range': Original intensity range [min, max]
                - 'normalized_range': Normalized intensity range [min, max]

        Raises:
            RuntimeError: If fit_population() has not been called
            FileNotFoundError: If input file does not exist
            RuntimeError: If normalization fails
        """
        # Check if normalizer has been fitted
        if self.fitted_normalizer is None:
            raise RuntimeError(
                "LSQ normalizer must be fitted before execute(). "
                "Call fit_population() first with a list of image paths."
            )

        allow_overwrite = kwargs.get("allow_overwrite", False)
        mask_path = kwargs.get("mask_path", None)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI with nibabel
            self.logger.debug(f"Loading image: {input_path}")
            input_img = nib.load(str(input_path))
            input_data = input_img.get_fdata()

            # Store original range
            original_range = [float(input_data.min()), float(input_data.max())]

            # Apply LSQ normalization using fitted normalizer
            self.logger.info("Applying LSQ normalization using fitted population parameters...")
            modality = infer_modality_from_filename(input_path)
            self.logger.info(f"Inferred modality: {modality}, input: {input_path}")

            # Use the fitted normalizer to transform this image
            normalized_data = self.fitted_normalizer(input_data, modality=modality)

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

            self.logger.info("LSQ normalization complete")

            return {
                "norm_value": self.norm_value,
                "fitted": True,
                "modality": str(modality),
                "original_range": original_range,
                "normalized_range": normalized_range,
            }

        except Exception as e:
            self.logger.error(f"LSQ normalization failed: {e}")
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
        - Metadata text with LSQ parameters and population fitting status

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters containing metadata:
                - 'norm_value': Normalization value
                - 'fitted': Whether normalizer was fitted
                - 'modality': Modality string
                - 'original_range': Original intensity range
                - 'normalized_range': Normalized intensity range

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        norm_value = kwargs.get("norm_value", self.norm_value)
        fitted = kwargs.get("fitted", False)
        modality = kwargs.get("modality", "unknown")
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
                f'LSQ Population Normalization: {before_path.stem}',
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
            axes[0, 3].hist(before_nonzero, bins=100, alpha=0.7, color='blue', density=True, label='Histogram')
            axes[0, 3].set_xlabel('Intensity', fontsize=10)
            axes[0, 3].set_ylabel('Density', fontsize=10)
            axes[0, 3].set_title('Original Histogram', fontsize=12)
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
            axes[1, 3].set_xlabel('Intensity', fontsize=10)
            axes[1, 3].set_ylabel('Density', fontsize=10)
            axes[1, 3].set_title('Normalized Histogram', fontsize=12)
            axes[1, 3].grid(True, alpha=0.3)

            # Add metadata text
            fitted_status = "Population-Fitted" if fitted else "Not Fitted (ERROR)"
            metadata_text = (
                f"Normalization Method: LSQ (Least-Squares Population)\n"
                f"  Status: {fitted_status}\n"
                f"  Modality: {modality}\n"
                f"  Norm_Value = {norm_value:.3f}\n\n"
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
