"""Otsu-based foreground extraction for background removal.

This module implements a robust background removal method using Otsu's threshold
and connected component analysis. Adapted from the js-ddpm-epilepsy project.

Algorithm:
1. Robust percentile scaling (p1, p99) to normalize intensities
2. Gaussian smoothing to reduce noise
3. Otsu thresholding (pure NumPy implementation, no sklearn dependency)
4. Connected component analysis to find largest foreground component(s)
5. Hole filling with binary_fill_holes

The method is designed to preserve anatomical structures while removing
air/background voxels effectively.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    generate_binary_structure,
    label,
)

from mengrowth.preprocessing.src.data_harmonization.base import BaseBackgroundRemover
from mengrowth.preprocessing.src.config import BackgroundZeroingConfig

logger = logging.getLogger(__name__)


def otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold on 1D data array.

    Implements Otsu's method to find the optimal threshold that minimizes
    intra-class variance between foreground and background.

    Args:
        data: 1D array of values to threshold.

    Returns:
        Optimal threshold value.
    """
    # Flatten and remove NaN/Inf
    data = data.ravel()
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return 0.5

    # Compute histogram
    n_bins = 256
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Total pixels
    total = hist.sum()
    if total == 0:
        return 0.5

    # Precompute cumulative sums
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg

    # Cumulative mean
    sum_bg = np.cumsum(hist * bin_centers)
    sum_fg = sum_bg[-1] - sum_bg

    # Calculate means
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = sum_fg / weight_fg

    # Calculate between-class variance
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Find threshold that maximizes between-class variance
    idx = np.nanargmax(variance)
    threshold = bin_centers[idx]

    return float(threshold)


class OtsuForegroundRemover(BaseBackgroundRemover):
    """Remove background using Otsu-based foreground extraction.

    This implements a robust method for background removal:
    1. Robust percentile scaling to normalize intensities
    2. Optional Gaussian smoothing
    3. Otsu thresholding to separate foreground/background
    4. Connected component analysis to keep largest component(s)
    5. Hole filling for clean masks
    6. Optional morphological operations for conservativeness control

    The method processes the entire 3D volume at once, making it robust
    to slice-to-slice variations.
    """

    def __init__(self, config: BackgroundZeroingConfig, verbose: bool = False) -> None:
        """Initialize Otsu foreground remover.

        Args:
            config: BackgroundZeroingConfig with method parameters
            verbose: Enable verbose logging
        """
        config_dict = {
            "method": config.method,
            "gaussian_sigma_px": getattr(config, "gaussian_sigma_px", 1.0),
            "min_component_voxels": getattr(config, "min_component_voxels", 1000),
            "n_components_to_keep": getattr(config, "n_components_to_keep", 1),
            "relaxed_threshold_factor": getattr(config, "relaxed_threshold_factor", 0.1),
            "p_low": getattr(config, "p_low", 1.0),
            "p_high": getattr(config, "p_high", 99.0),
            "fill_value": getattr(config, "fill_value", 0.0),
            "air_border_margin": getattr(config, "air_border_margin", 0),
            "expand_air_mask": getattr(config, "expand_air_mask", 0),
        }
        super().__init__(config=config_dict, verbose=verbose)
        self.bg_config = config

        # Extract parameters with fallbacks
        self.gaussian_sigma_px = config_dict["gaussian_sigma_px"]
        self.min_component_voxels = config_dict["min_component_voxels"]
        self.n_components_to_keep = config_dict["n_components_to_keep"]
        self.relaxed_threshold_factor = config_dict["relaxed_threshold_factor"]
        self.p_low = config_dict["p_low"]
        self.p_high = config_dict["p_high"]
        self.fill_value = config_dict["fill_value"]
        self.air_border_margin = config_dict["air_border_margin"]
        self.expand_air_mask = config_dict["expand_air_mask"]

        self.logger.info(
            f"Initialized OtsuForegroundRemover: "
            f"sigma={self.gaussian_sigma_px}px, "
            f"min_voxels={self.min_component_voxels}, "
            f"n_components={self.n_components_to_keep}, "
            f"percentiles=[{self.p_low}, {self.p_high}]"
        )

    def _compute_foreground_mask(self, volume: np.ndarray) -> np.ndarray:
        """Compute foreground mask using Otsu-based method.

        Args:
            volume: 3D MRI volume array

        Returns:
            Boolean mask where True=foreground (brain/head), False=background (air)
        """
        self.logger.info("Computing foreground mask using Otsu method...")

        # Step 1: Robust percentile scaling
        p_low_val = np.percentile(volume, self.p_low)
        p_high_val = np.percentile(volume, self.p_high)

        if p_high_val <= p_low_val:
            self.logger.warning(
                f"Invalid percentile range: [{p_low_val:.4f}, {p_high_val:.4f}]. "
                "Returning full volume as foreground."
            )
            return np.ones_like(volume, dtype=bool)

        # Scale to [0, 1]
        scaled = np.clip(
            (volume.astype(np.float64) - p_low_val) / (p_high_val - p_low_val),
            0,
            1
        )
        self.logger.debug(f"Percentile scaling: [{p_low_val:.4f}, {p_high_val:.4f}] -> [0, 1]")

        # Step 2: Gaussian smoothing
        if self.gaussian_sigma_px > 0:
            self.logger.debug(f"Applying Gaussian smoothing with sigma={self.gaussian_sigma_px}px")
            smoothed = gaussian_filter(scaled, sigma=self.gaussian_sigma_px)
        else:
            smoothed = scaled

        # Step 3: Otsu thresholding
        threshold = otsu_threshold(smoothed)
        self.logger.info(f"Otsu threshold: {threshold:.4f}")

        binary = smoothed > threshold
        n_foreground = binary.sum()
        self.logger.debug(
            f"Initial foreground: {n_foreground} voxels "
            f"({100 * n_foreground / binary.size:.2f}%)"
        )

        # Step 4: Connected component analysis
        struct = generate_binary_structure(3, 2)  # 18-connectivity
        labeled, n_components = label(binary, structure=struct)

        if n_components == 0:
            self.logger.warning("No components found - returning empty foreground mask")
            return np.zeros_like(volume, dtype=bool)

        self.logger.debug(f"Found {n_components} connected components")

        # Step 5: Find and keep largest component(s)
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background label

        # Sort indices by size (descending)
        sorted_indices = np.argsort(component_sizes)[::-1]

        # Keep top N components
        n_to_keep = min(self.n_components_to_keep, n_components)
        keep_indices = []

        for idx in sorted_indices:
            if idx == 0:  # Skip background
                continue

            # First component: apply full threshold
            if len(keep_indices) == 0:
                if component_sizes[idx] >= self.min_component_voxels:
                    keep_indices.append(idx)
                    self.logger.debug(
                        f"Keeping primary component {idx}: "
                        f"{component_sizes[idx]} voxels"
                    )
            else:
                # Additional components: use relaxed threshold
                relaxed_threshold = self.min_component_voxels * self.relaxed_threshold_factor
                if component_sizes[idx] >= relaxed_threshold:
                    keep_indices.append(idx)
                    self.logger.debug(
                        f"Keeping secondary component {idx}: "
                        f"{component_sizes[idx]} voxels (relaxed threshold)"
                    )

            if len(keep_indices) >= n_to_keep:
                break

        if len(keep_indices) == 0:
            self.logger.warning(
                f"No components larger than {self.min_component_voxels} voxels. "
                "Returning largest component regardless of size."
            )
            # Fallback: keep largest component regardless of size
            largest_idx = sorted_indices[0] if sorted_indices[0] != 0 else sorted_indices[1]
            keep_indices = [largest_idx]

        # Create foreground mask with all kept components
        foreground_mask = np.zeros_like(binary, dtype=bool)
        for idx in keep_indices:
            foreground_mask |= (labeled == idx)

        # Step 6: Fill holes
        self.logger.debug("Filling holes in foreground mask")
        foreground_mask = binary_fill_holes(foreground_mask)

        # Step 7: Apply morphological operations for conservativeness control
        struct = generate_binary_structure(3, 1)  # 6-connectivity

        # Dilate foreground to be MORE conservative (include more as foreground)
        if self.air_border_margin > 0:
            self.logger.debug(
                f"Dilating foreground by {self.air_border_margin} voxels "
                "(MORE conservative - protects anatomy)"
            )
            for _ in range(self.air_border_margin):
                foreground_mask = binary_dilation(foreground_mask, structure=struct)

        # Erode foreground to be LESS conservative (remove more background)
        if self.expand_air_mask > 0:
            self.logger.debug(
                f"Eroding foreground by {self.expand_air_mask} voxels "
                "(LESS conservative - removes more background)"
            )
            for _ in range(self.expand_air_mask):
                foreground_mask = binary_erosion(foreground_mask, structure=struct)

        final_size = foreground_mask.sum()
        self.logger.info(
            f"Final foreground mask: {final_size} voxels "
            f"({100 * final_size / foreground_mask.size:.2f}%)"
        )

        return foreground_mask

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Remove background from NIfTI volume using Otsu method.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output NIfTI file
            **kwargs: Additional parameters (allow_overwrite)

        Returns:
            Dict with execution metadata:
                - foreground_voxels: Number of foreground voxels
                - background_voxels: Number of background voxels
                - foreground_fraction: Fraction of volume as foreground
                - otsu_threshold: Computed Otsu threshold

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If background removal fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI
            img = nib.load(str(input_path))
            volume = img.get_fdata()

            self.logger.info(
                f"Loaded volume: shape={volume.shape}, "
                f"range=[{volume.min():.4f}, {volume.max():.4f}]"
            )

            # Compute foreground mask
            foreground_mask = self._compute_foreground_mask(volume)

            # Apply mask: set background voxels to fill_value
            masked_volume = volume.copy()
            masked_volume[~foreground_mask] = self.fill_value

            n_background = (~foreground_mask).sum()
            n_foreground = foreground_mask.sum()
            self.logger.info(
                f"Set {n_background} background voxels to {self.fill_value} "
                f"({100 * n_background / foreground_mask.size:.2f}%)"
            )

            # Create new NIfTI image with same header
            output_img = nib.Nifti1Image(
                masked_volume.astype(np.float32),
                affine=img.affine,
                header=img.header
            )

            # Save output
            nib.save(output_img, str(output_path))

            self.logger.info(f"Successfully removed background from {input_path.name}")

            return {
                "foreground_voxels": int(n_foreground),
                "background_voxels": int(n_background),
                "foreground_fraction": float(n_foreground / foreground_mask.size),
                "method": "otsu_foreground",
            }

        except Exception as e:
            self.logger.error(f"Background removal failed: {e}")
            raise RuntimeError(f"Background removal failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization showing background removal with mask overlay.

        Creates a figure with 3 rows (axial, sagittal, coronal) x 4 columns
        showing depth slices with the foreground mask overlaid in green.

        Args:
            before_path: Path to original NIfTI file
            after_path: Path to processed NIfTI file
            output_path: Path to save visualization PNG
            **kwargs: Additional parameters (not used)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.logger.info(f"Generating Otsu background removal visualization: {output_path}")

        try:
            # Load before and after
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Compute foreground mask from the before volume
            foreground_mask = self._compute_foreground_mask(before_data)

            # Log statistics
            changed_to_zero = (before_data != after_data) & (after_data == self.fill_value)
            self.logger.info(
                f"Visualization: foreground mask={foreground_mask.sum()} voxels "
                f"({100 * foreground_mask.sum() / foreground_mask.size:.2f}%), "
                f"newly zeroed={changed_to_zero.sum()} voxels"
            )

            # Get 4 depth slices for visualization
            depth_fractions = [0.25, 0.4, 0.5, 0.6]

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(
                f'Otsu Background Removal: {before_path.stem}\n'
                f'Foreground mask overlaid (green=kept, transparent=removed)',
                fontsize=14,
                fontweight='bold'
            )

            # Get intensity range for consistent display
            vmin = np.percentile(before_data[before_data > 0], 1)
            vmax = np.percentile(before_data[before_data > 0], 99)

            # Axial slices (row 0)
            for col, frac in enumerate(depth_fractions):
                z_idx = int(before_data.shape[2] * frac)
                slice_data = before_data[:, :, z_idx].T
                slice_mask = foreground_mask[:, :, z_idx].T

                axes[0, col].imshow(slice_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                # Overlay foreground mask in green
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [0, 1, 0, 0.2]  # green with alpha
                axes[0, col].imshow(overlay, origin='lower')
                axes[0, col].set_title(f'Axial z={z_idx}', fontsize=10)
                axes[0, col].axis('off')

            # Sagittal slices (row 1)
            for col, frac in enumerate(depth_fractions):
                x_idx = int(before_data.shape[0] * frac)
                slice_data = before_data[x_idx, :, :].T
                slice_mask = foreground_mask[x_idx, :, :].T

                axes[1, col].imshow(slice_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [0, 1, 0, 0.2]
                axes[1, col].imshow(overlay, origin='lower')
                axes[1, col].set_title(f'Sagittal x={x_idx}', fontsize=10)
                axes[1, col].axis('off')

            # Coronal slices (row 2)
            for col, frac in enumerate(depth_fractions):
                y_idx = int(before_data.shape[1] * frac)
                slice_data = before_data[:, y_idx, :].T
                slice_mask = foreground_mask[:, y_idx, :].T

                axes[2, col].imshow(slice_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [0, 1, 0, 0.2]
                axes[2, col].imshow(overlay, origin='lower')
                axes[2, col].set_title(f'Coronal y={y_idx}', fontsize=10)
                axes[2, col].axis('off')

            # Add statistics text
            fg_pct = 100 * foreground_mask.sum() / foreground_mask.size
            stats_text = (
                f"Foreground: {foreground_mask.sum():,} voxels ({fg_pct:.1f}%)\n"
                f"Sigma: {self.gaussian_sigma_px}px | "
                f"Min component: {self.min_component_voxels} voxels"
            )
            fig.text(
                0.5, 0.02, stats_text,
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
