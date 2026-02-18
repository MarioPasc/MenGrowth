"""Z-score normalization on brain voxels.

This module implements z-score intensity normalization that standardizes images
by subtracting the mean and dividing by the standard deviation, computed on
brain voxels only (excluding background zeros from skull-stripped images).

This implements the nnU-Net/BraTS convention:
    I'(x) = (I(x) - mean_brain) / std_brain

where mean_brain and std_brain are computed over brain voxels only (mask > 0).
Background voxels remain exactly 0.

Reference:
    nnU-Net: Isensee et al. (2021), "nnU-Net: a self-configuring method for
    deep learning-based biomedical image segmentation"
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer

logger = logging.getLogger(__name__)


class ZScoreNormalizer(BaseNormalizer):
    """Z-score intensity normalizer using intensity-normalization package.

    This normalizer applies voxel-wise z-score normalization by subtracting the mean
    intensity and dividing by the standard deviation:

        I'(x) = (I(x) - μ) / σ

    where μ is the mean and σ is the standard deviation of in-mask intensities.
    The result is then optionally scaled by a normalization value.

    This uses the intensity-normalization package's normalize_image function with method="zscore".
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize z-score normalizer.

        Args:
            config: Configuration dictionary containing:
                - norm_value: Scaling factor after z-score (default=1.0)
                - clip_range: Optional [low, high] clipping bounds after z-score (default=None)
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)

        # Extract norm_value parameter with default
        self.norm_value = config.get("norm_value", 1.0)
        self.clip_range: Optional[List[float]] = config.get("clip_range", None)

        if self.norm_value <= 0:
            raise ValueError(f"norm_value must be positive, got {self.norm_value}")

        if self.clip_range is not None:
            if len(self.clip_range) != 2 or self.clip_range[0] >= self.clip_range[1]:
                raise ValueError(
                    f"clip_range must be [low, high] with low < high, got {self.clip_range}"
                )

        self.logger.info(
            f"Initialized ZScoreNormalizer: norm_value={self.norm_value}, clip_range={self.clip_range}"
        )

    def execute(
        self, input_path: Path, output_path: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute z-score normalization using intensity-normalization package.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - mask_path: Optional path to brain mask (Path)

        Returns:
            Dictionary containing:
                - 'mean': Mean intensity value
                - 'std': Standard deviation
                - 'norm_value': Normalization scaling factor
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
            # Load NIfTI with nibabel
            self.logger.debug(f"Loading image: {input_path}")
            input_img = nib.load(str(input_path))
            input_data = input_img.get_fdata()

            # Store original range
            original_range = [float(input_data.min()), float(input_data.max())]

            # Load or derive brain mask
            if mask_path is not None and Path(mask_path).exists():
                mask_img = nib.load(str(mask_path))
                brain_mask = mask_img.get_fdata() > 0
                mask_source = "skull_stripping"
                self.logger.info(f"Using brain mask from: {mask_path}")
            else:
                brain_mask = input_data > 0
                mask_source = "nonzero_fallback"
                self.logger.info(
                    "No brain mask provided, using nonzero voxels as fallback"
                )

            # Defensive shape check: resample mask if shape doesn't match image
            if brain_mask.shape != input_data.shape:
                self.logger.warning(
                    f"Brain mask shape {brain_mask.shape} != image shape {input_data.shape}, resampling mask"
                )
                from scipy.ndimage import zoom

                factors = tuple(
                    s_i / s_m for s_i, s_m in zip(input_data.shape, brain_mask.shape)
                )
                brain_mask = zoom(brain_mask.astype(np.float32), factors, order=3) > 0.5

            brain_voxels = input_data[brain_mask]
            total_voxels = input_data.size
            brain_voxel_count = int(brain_voxels.size)
            brain_coverage_percent = 100.0 * brain_voxel_count / total_voxels

            self.logger.info(
                f"Brain coverage: {brain_coverage_percent:.1f}% ({brain_voxel_count}/{total_voxels} voxels)"
            )

            # Compute mean and std on brain voxels only (nnU-Net/BraTS convention)
            mean_val = float(np.mean(brain_voxels))
            std_val = float(np.std(brain_voxels))

            self.logger.info(
                f"Brain voxel stats: mean={mean_val:.3f}, std={std_val:.3f}"
            )

            if std_val < 1e-8:
                self.logger.warning(
                    f"Standard deviation is near zero ({std_val:.3e}). "
                    "Z-score normalization may not be meaningful."
                )
                std_val = 1e-8

            # Apply z-score: normalized[brain_mask] = (image[brain_mask] - mean) / std
            # Background stays 0
            normalized_data = np.zeros_like(input_data, dtype=np.float64)
            normalized_data[brain_mask] = (input_data[brain_mask] - mean_val) / std_val

            # Apply optional clipping
            if self.clip_range is not None:
                low, high = self.clip_range
                normalized_data[brain_mask] = np.clip(
                    normalized_data[brain_mask], low, high
                )
                self.logger.info(f"Clipped z-score values to [{low}, {high}]")

            # Store normalized range (brain voxels only)
            brain_normalized = normalized_data[brain_mask]
            normalized_range = [
                float(brain_normalized.min()),
                float(brain_normalized.max()),
            ]

            self.logger.info(
                f"Original range: [{original_range[0]:.3f}, {original_range[1]:.3f}]"
            )
            self.logger.info(
                f"Normalized range (brain): [{normalized_range[0]:.3f}, {normalized_range[1]:.3f}]"
            )

            # Save normalized image
            self.logger.debug(f"Saving normalized image: {output_path}")
            nib.Nifti1Image(normalized_data, input_img.affine).to_filename(
                str(output_path)
            )

            self.logger.info("Z-score normalization complete")

            return {
                "mean": mean_val,
                "std": std_val,
                "norm_value": self.norm_value,
                "clip_range": self.clip_range,
                "original_range": original_range,
                "normalized_range": normalized_range,
                "brain_voxel_count": brain_voxel_count,
                "brain_coverage_percent": brain_coverage_percent,
                "mask_source": mask_source,
            }

        except Exception as e:
            self.logger.error(f"Z-score normalization failed: {e}")
            raise RuntimeError(f"Normalization failed: {e}") from e

    def visualize(
        self, before_path: Path, after_path: Path, output_path: Path, **kwargs: Any
    ) -> None:
        """Generate visualization comparing before and after normalization.

        Creates visualization with:
        - Before and after image slices (axial, sagittal, coronal)
        - Before and after intensity histograms
        - Metadata text with mean, std, and ranges

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters containing metadata:
                - 'mean': Mean intensity value
                - 'std': Standard deviation
                - 'original_range': Original intensity range
                - 'normalized_range': Normalized intensity range

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean_value = kwargs.get("mean")
        std_value = kwargs.get("std")
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
                f"Z-Score Normalization: {before_path.stem}",
                fontsize=16,
                fontweight="bold",
            )

            # Row 1: Original image
            # Compute intensity range for original (for consistent display)
            vmin_before = before_data.min()
            vmax_before = before_data.max()

            axes[0, 0].imshow(
                axial_before,
                cmap="gray",
                origin="lower",
                vmin=vmin_before,
                vmax=vmax_before,
            )
            axes[0, 0].set_title("Original - Axial", fontsize=12)
            axes[0, 0].axis("off")

            axes[0, 1].imshow(
                sagittal_before,
                cmap="gray",
                origin="lower",
                vmin=vmin_before,
                vmax=vmax_before,
            )
            axes[0, 1].set_title("Original - Sagittal", fontsize=12)
            axes[0, 1].axis("off")

            axes[0, 2].imshow(
                coronal_before,
                cmap="gray",
                origin="lower",
                vmin=vmin_before,
                vmax=vmax_before,
            )
            axes[0, 2].set_title("Original - Coronal", fontsize=12)
            axes[0, 2].axis("off")

            # Histogram for original
            before_nonzero = before_data[before_data > 0]
            axes[0, 3].hist(
                before_nonzero, bins=100, alpha=0.7, color="blue", density=True
            )
            axes[0, 3].axvline(
                mean_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={mean_value:.2f}",
            )
            axes[0, 3].axvline(
                mean_value - std_value,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="μ-σ",
            )
            axes[0, 3].axvline(
                mean_value + std_value,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="μ+σ",
            )
            axes[0, 3].set_xlabel("Intensity", fontsize=10)
            axes[0, 3].set_ylabel("Density", fontsize=10)
            axes[0, 3].set_title("Original Histogram", fontsize=12)
            axes[0, 3].legend(fontsize=8)
            axes[0, 3].grid(True, alpha=0.3)

            # Row 2: Normalized image
            # Compute intensity range for normalized
            vmin_after = after_data.min()
            vmax_after = after_data.max()

            axes[1, 0].imshow(
                axial_after,
                cmap="gray",
                origin="lower",
                vmin=vmin_after,
                vmax=vmax_after,
            )
            axes[1, 0].set_title("Normalized - Axial", fontsize=12)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(
                sagittal_after,
                cmap="gray",
                origin="lower",
                vmin=vmin_after,
                vmax=vmax_after,
            )
            axes[1, 1].set_title("Normalized - Sagittal", fontsize=12)
            axes[1, 1].axis("off")

            axes[1, 2].imshow(
                coronal_after,
                cmap="gray",
                origin="lower",
                vmin=vmin_after,
                vmax=vmax_after,
            )
            axes[1, 2].set_title("Normalized - Coronal", fontsize=12)
            axes[1, 2].axis("off")

            # Histogram for normalized
            after_nonzero = after_data[after_data > vmin_after]
            axes[1, 3].hist(
                after_nonzero, bins=100, alpha=0.7, color="green", density=True
            )
            axes[1, 3].axvline(
                0, color="red", linestyle="--", linewidth=2, label="Mean≈0"
            )
            axes[1, 3].axvline(
                -self.norm_value,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="±σ",
            )
            axes[1, 3].axvline(
                self.norm_value, color="orange", linestyle=":", linewidth=1.5
            )
            axes[1, 3].set_xlabel("Intensity", fontsize=10)
            axes[1, 3].set_ylabel("Density", fontsize=10)
            axes[1, 3].set_title("Normalized Histogram", fontsize=12)
            axes[1, 3].legend(fontsize=8)
            axes[1, 3].grid(True, alpha=0.3)

            # Add metadata text
            metadata_text = (
                f"Normalization Method: Z-Score\n"
                f"  Mean (μ) = {mean_value:.3f}\n"
                f"  Std Dev (σ) = {std_value:.3f}\n"
                f"  Norm Value = {self.norm_value:.3f}\n\n"
                f"Original Range: [{original_range[0]:.3f}, {original_range[1]:.3f}]\n"
                f"Normalized Range: [{normalized_range[0]:.3f}, {normalized_range[1]:.3f}]"
            )

            fig.text(
                0.5,
                0.01,
                metadata_text,
                ha="center",
                fontsize=10,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout(rect=[0, 0.08, 1, 0.98])

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
