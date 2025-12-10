"""FCM-based normalization using intensity-normalization package.

This module implements Fuzzy C-Means (FCM) clustering-based intensity normalization
which segments the brain into tissue types (CSF, GM, WM) and normalizes based on
the target tissue's mean intensity. This is particularly useful for standardizing
MRI intensities based on tissue-specific characteristics.

Reference:
    intensity-normalization package: https://github.com/jcreinhold/intensity-normalization
    FCM paper: Bezdek et al. (1984), "FCM: The fuzzy c-means clustering algorithm"
"""

from pathlib import Path
from typing import Any, Dict
import logging

import nibabel as nib
import numpy as np

from intensity_normalization.normalizers.individual.fcm import FCMNormalizer as FCMNormalize
from intensity_normalization.domain.models import TissueType

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer
from mengrowth.preprocessing.src.normalization.utils import infer_modality_from_filename

logger = logging.getLogger(__name__)


def tissue_type_str_to_enum(tissue_str: str) -> TissueType:
    """Convert tissue type string to TissueType enum.

    Args:
        tissue_str: String representation of tissue type ("WM", "GM", "CSF")

    Returns:
        TissueType enum value

    Raises:
        ValueError: If tissue_str is not a valid tissue type
    """
    tissue_map = {
        "wm": TissueType.WM,
        "gm": TissueType.GM,
        "csf": TissueType.CSF,
    }

    tissue_lower = tissue_str.lower()
    if tissue_lower not in tissue_map:
        raise ValueError(
            f"Unsupported tissue type: {tissue_str}. "
            f"Must be one of: WM, GM, CSF"
        )

    return tissue_map[tissue_lower]


class FCMNormalizer(BaseNormalizer):
    """Fuzzy C-Means tissue-based intensity normalizer.

    This normalizer uses fuzzy C-means clustering to identify tissue types
    (CSF, gray matter, white matter) and normalizes intensities by dividing
    by the mean intensity of the target tissue type.

    This uses the intensity-normalization package's FCMNormalizer.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize FCM normalizer.

        Args:
            config: Configuration dictionary containing:
                - n_clusters: Number of tissue clusters (default=3)
                - tissue_type: Target tissue type string "WM"|"GM"|"CSF" (default="WM")
                - max_iter: Maximum FCM iterations (default=50)
                - error_threshold: Convergence threshold (default=0.005)
                - fuzziness: Cluster membership fuzziness parameter (default=2.0)
            verbose: Enable verbose logging
        """
        super().__init__(
            config=config,
            verbose=verbose
        )

        # Extract parameters with defaults
        self.n_clusters = config.get("n_clusters", 3)
        tissue_type_str = config.get("tissue_type", "WM")
        self.max_iter = config.get("max_iter", 50)
        self.error_threshold = config.get("error_threshold", 0.005)
        self.fuzziness = config.get("fuzziness", 2.0)

        # Convert tissue type string to enum
        self.tissue_type = tissue_type_str_to_enum(tissue_type_str)
        self.tissue_type_str = tissue_type_str.upper()  # Store string for logging

        # Validate parameters
        if not 2 <= self.n_clusters <= 10:
            raise ValueError(f"n_clusters must be in [2, 10], got {self.n_clusters}")

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")

        if not 0.0 < self.error_threshold < 1.0:
            raise ValueError(
                f"error_threshold must be in (0.0, 1.0), got {self.error_threshold}"
            )

        if not 1.0 < self.fuzziness <= 10.0:
            raise ValueError(
                f"fuzziness must be in (1.0, 10.0], got {self.fuzziness}"
            )

        self.logger.info(
            f"Initialized FCMNormalizer: n_clusters={self.n_clusters}, "
            f"tissue_type={self.tissue_type_str}, max_iter={self.max_iter}, "
            f"error_threshold={self.error_threshold}, fuzziness={self.fuzziness}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute FCM-based normalization using intensity-normalization package.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - mask_path: Optional path to brain mask (Path)

        Returns:
            Dictionary containing:
                - 'n_clusters': Number of clusters used
                - 'tissue_type': Target tissue type string
                - 'max_iter': Maximum iterations
                - 'error_threshold': Convergence threshold
                - 'fuzziness': Fuzziness parameter
                - 'modality': Inferred modality
                - 'original_range': Original intensity range [min, max]
                - 'normalized_range': Normalized intensity range [min, max]

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
            # Load NIfTI with nibabel
            self.logger.debug(f"Loading image: {input_path}")
            input_img = nib.load(str(input_path))
            
            input_img.get_data = input_img.get_fdata  # For compatibility with older nibabel versions
            input_img.with_data = lambda data: nib.Nifti1Image(data, input_img.affine)
            
            # Apply FCM normalization using intensity-normalization package
            self.logger.info("Applying FCM normalization using intensity-normalization package...")
            normalizer = FCMNormalize(
                n_clusters=self.n_clusters,
                tissue_type=self.tissue_type,
                max_iter=self.max_iter,
                error_threshold=self.error_threshold,
                fuzziness=self.fuzziness
            )
            modality = infer_modality_from_filename(str(input_path))
            self.logger.info(f"Inferred modality: {modality}, input: {input_path}")
            
            # Pass nibabel image to normalizer
            normalized_result = normalizer(input_img)

            if type(input_img) is not np.ndarray:
                input_data = input_img.get_fdata()
            else:
                input_data = input_img

            # Store original range
            original_range = [float(input_data.min()), float(input_data.max())]

            # Extract data from result
            if hasattr(normalized_result, 'get_fdata'):
                normalized_data = normalized_result.get_fdata()
            elif isinstance(normalized_result, np.ndarray):
                normalized_data = normalized_result
            else:
                # Fallback
                normalized_data = normalized_result.get_fdata()

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

            self.logger.info("FCM normalization complete")

            return {
                "n_clusters": self.n_clusters,
                "tissue_type": self.tissue_type_str,
                "max_iter": self.max_iter,
                "error_threshold": self.error_threshold,
                "fuzziness": self.fuzziness,
                "modality": str(modality),
                "original_range": original_range,
                "normalized_range": normalized_range,
            }

        except Exception as e:
            self.logger.error(f"FCM normalization failed: {e}")
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
        - Metadata text with FCM parameters and ranges

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters containing metadata:
                - 'n_clusters': Number of clusters
                - 'tissue_type': Target tissue type
                - 'max_iter': Maximum iterations
                - 'error_threshold': Convergence threshold
                - 'fuzziness': Fuzziness parameter
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

        n_clusters = kwargs.get("n_clusters", self.n_clusters)
        tissue_type = kwargs.get("tissue_type", self.tissue_type_str)
        max_iter = kwargs.get("max_iter", self.max_iter)
        error_threshold = kwargs.get("error_threshold", self.error_threshold)
        fuzziness = kwargs.get("fuzziness", self.fuzziness)
        modality = kwargs.get("modality", "unknown")
        original_range = kwargs.get("original_range")
        normalized_range = kwargs.get("normalized_range")

        self.logger.info(f"Generating normalization visualization: {output_path}")

        try:
            # Load images
            before_img = nib.load(str(before_path))
            after_img = nib.load(str(after_path))
            

            if type(before_img) is not np.ndarray:
                before_data = before_img.get_fdata()
            else:
                before_data = before_img
            
            if type(after_img) is not np.ndarray:
                after_data = after_img.get_fdata()
            else:
                after_data = after_img

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
                f'FCM Normalization: {before_path.stem}',
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
            metadata_text = (
                f"Normalization Method: FCM (Fuzzy C-Means)\n"
                f"  Modality: {modality}\n"
                f"  Target Tissue: {tissue_type}\n"
                f"  N_Clusters = {n_clusters}\n"
                f"  Max_Iter = {max_iter}\n"
                f"  Error_Threshold = {error_threshold:.4f}\n"
                f"  Fuzziness = {fuzziness:.2f}\n\n"
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
