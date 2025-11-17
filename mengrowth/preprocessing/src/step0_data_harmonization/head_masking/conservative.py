"""Background removal operations for data harmonization.

This module implements conservative background removal that sets only air/background
voxels to zero without eroding anatomical structures. This is NOT skull-stripping.
"""

from pathlib import Path
from typing import Any, Dict
import logging
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_erosion, binary_dilation, generate_binary_structure

from mengrowth.preprocessing.src.step0_data_harmonization.base import BaseBackgroundRemover
from mengrowth.preprocessing.src.config import BackgroundZeroingConfig

logger = logging.getLogger(__name__)


class ConservativeBackgroundRemover(BaseBackgroundRemover):
    """Remove background/air voxels conservatively without eroding anatomy.

    This implements the "border_connected_percentile" method specified in the brief:
    1. Compute low-percentile intensity threshold on full volume
    2. Optionally smooth volume before thresholding
    3. Apply 3D connected components analysis
    4. Keep largest exterior component touching volume borders as "air"
    5. Optionally erode air mask to be extra conservative
    6. Set only air voxels to zero, leave everything else unchanged

    The method is designed to be conservative: prefer under-masking (leaving some
    background) to over-masking (eroding anatomy).
    """

    def __init__(self, config: BackgroundZeroingConfig, verbose: bool = False) -> None:
        """Initialize conservative background remover.

        Args:
            config: BackgroundZeroingConfig with method parameters
            verbose: Enable verbose logging
        """
        # Convert dataclass to dict for base class
        config_dict = {
            "method": config.method,
            "percentile_low": config.percentile_low,
            "gaussian_sigma": config.gaussian_sigma,
            "min_comp_voxels": config.min_comp_voxels,
            "air_border_margin": config.air_border_margin,
        }
        super().__init__(config=config_dict, verbose=verbose)
        self.bg_config = config
        self.logger.info(
            f"Initialized ConservativeBackgroundRemover: "
            f"method={config.method}, percentile={config.percentile_low}%, "
            f"sigma={config.gaussian_sigma}, air_margin={config.air_border_margin}px"
        )

    def _build_air_mask(self, volume: np.ndarray) -> np.ndarray:
        """Build conservative air/background mask using border-connected percentile method.

        Args:
            volume: 3D MRI volume array

        Returns:
            Boolean mask where True=air, False=anatomy/foreground
        """
        self.logger.info("Building conservative air mask...")

        # Step 1: Optionally smooth volume before thresholding
        if self.bg_config.gaussian_sigma > 0:
            self.logger.debug(f"Smoothing volume with sigma={self.bg_config.gaussian_sigma}")
            smoothed = gaussian_filter(volume.astype(np.float32), sigma=self.bg_config.gaussian_sigma)
        else:
            smoothed = volume.astype(np.float32)

        # Step 2: Compute low percentile threshold
        threshold = np.percentile(smoothed, self.bg_config.percentile_low)
        self.logger.info(f"Low percentile ({self.bg_config.percentile_low}%) threshold: {threshold:.4f}")

        # Step 3: Classify candidate background voxels
        candidate_air = smoothed <= threshold
        n_candidate = candidate_air.sum()
        self.logger.debug(f"Candidate air voxels: {n_candidate} ({100 * n_candidate / candidate_air.size:.2f}%)")

        # Step 4: Apply 3D connected components
        struct = generate_binary_structure(3, 3)  # 26-connectivity
        labeled, num_components = label(candidate_air, structure=struct)
        self.logger.debug(f"Found {num_components} connected components")

        if num_components == 0:
            self.logger.warning("No components found - returning empty air mask")
            return np.zeros_like(volume, dtype=bool)

        # Step 5: Find components touching volume borders (exterior components)
        border_labels = set()

        # Check all 6 faces of the volume
        # X faces
        border_labels.update(labeled[0, :, :].flat)
        border_labels.update(labeled[-1, :, :].flat)
        # Y faces
        border_labels.update(labeled[:, 0, :].flat)
        border_labels.update(labeled[:, -1, :].flat)
        # Z faces
        border_labels.update(labeled[:, :, 0].flat)
        border_labels.update(labeled[:, :, -1].flat)

        # Remove background label (0)
        border_labels.discard(0)

        self.logger.debug(f"Found {len(border_labels)} components touching borders")

        # Step 6: Find largest border-connected component
        if not border_labels:
            self.logger.warning("No border-connected components found - returning empty air mask")
            return np.zeros_like(volume, dtype=bool)

        # Count voxels in each border component
        component_sizes = {}
        for lbl in border_labels:
            size = (labeled == lbl).sum()
            if size >= self.bg_config.min_comp_voxels:
                component_sizes[lbl] = size

        if not component_sizes:
            self.logger.warning(
                f"No border components larger than {self.bg_config.min_comp_voxels} voxels - "
                "returning empty air mask"
            )
            return np.zeros_like(volume, dtype=bool)

        # Select largest component as air
        largest_label = max(component_sizes, key=component_sizes.get)
        largest_size = component_sizes[largest_label]
        self.logger.info(
            f"Selected largest border component as air: "
            f"label={largest_label}, size={largest_size} voxels "
            f"({100 * largest_size / volume.size:.2f}%)"
        )

        air_mask = labeled == largest_label

        # Step 7a: Optionally erode air mask to be MORE conservative
        # (shrink the air region, not the anatomy)
        if self.bg_config.air_border_margin > 0:
            self.logger.debug(f"Eroding air mask by {self.bg_config.air_border_margin} voxels (MORE conservative)")
            erosion_struct = generate_binary_structure(3, 1)  # 6-connectivity
            for _ in range(self.bg_config.air_border_margin):
                air_mask = binary_erosion(air_mask, structure=erosion_struct)

            eroded_size = air_mask.sum()
            self.logger.debug(
                f"After erosion: {eroded_size} air voxels ({100 * eroded_size / volume.size:.2f}%)"
            )

        # Step 7b: Optionally dilate air mask to be LESS conservative
        # (expand the air region to remove more background)
        expand_param = getattr(self.bg_config, "expand_air_mask", 0)
        if expand_param > 0:
            self.logger.debug(f"Dilating air mask by {expand_param} voxels (LESS conservative)")
            dilation_struct = generate_binary_structure(3, 1)  # 6-connectivity
            for _ in range(expand_param):
                air_mask = binary_dilation(air_mask, structure=dilation_struct)

            dilated_size = air_mask.sum()
            self.logger.debug(
                f"After dilation: {dilated_size} air voxels ({100 * dilated_size / volume.size:.2f}%)"
            )

        final_size = air_mask.sum()
        self.logger.info(
            f"Final air mask: {final_size} voxels ({100 * final_size / volume.size:.2f}%)"
        )

        return air_mask

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Remove background from NIfTI volume conservatively.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output NIfTI file
            **kwargs: Additional parameters (allow_overwrite)

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

            # Build air mask
            air_mask = self._build_air_mask(volume)

            # Apply mask: set air voxels to zero
            masked_volume = volume.copy()
            masked_volume[air_mask] = 0.0

            n_zeroed = air_mask.sum()
            self.logger.info(
                f"Zeroed {n_zeroed} background voxels ({100 * n_zeroed / air_mask.size:.2f}%)"
            )

            # Create new NIfTI image with same header
            output_img = nib.Nifti1Image(masked_volume, affine=img.affine, header=img.header)

            # Save output
            nib.save(output_img, str(output_path))

            self.logger.info(f"Successfully removed background from {input_path.name}")

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

        Displays 4 depth slices for each of 3 orientations, with the computed
        air mask overlaid on the original volume.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.logger.info(f"Generating background removal visualization: {output_path}")

        try:
            # Load before and after
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Compute air mask directly from the "before" volume
            # (this is the same algorithm used in execute)
            air_mask = self._build_air_mask(before_data)

            # Optional: log how many voxels actually changed to zero
            changed_to_zero = (before_data != after_data) & (after_data == 0)
            self.logger.info(
                f"Visualization: air mask={air_mask.sum()} voxels "
                f"({100 * air_mask.sum() / air_mask.size:.2f}%), "
                f"newly zeroed={changed_to_zero.sum()} voxels "
                f"({100 * changed_to_zero.sum() / air_mask.size:.2f}%)"
            )

            # Get 4 depth slices for visualization
            depth_fractions = [0.25, 0.4, 0.5, 0.6]

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(
                f'Background Removal: {before_path.stem}\n'
                f'Air mask overlaid on original (red=air/background)',
                fontsize=16,
                fontweight='bold'
            )

            # Axial slices (row 0)
            for col, frac in enumerate(depth_fractions):
                z_idx = int(before_data.shape[2] * frac)
                slice_data = before_data[:, :, z_idx].T
                slice_mask = air_mask[:, :, z_idx].T

                axes[0, col].imshow(slice_data, cmap='gray', origin='lower')
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [1, 0, 0, 0.3]  # red with alpha
                axes[0, col].imshow(overlay, origin='lower')
                axes[0, col].set_title(f'Axial z={z_idx}')
                axes[0, col].axis('off')

            # Sagittal slices (row 1)
            for col, frac in enumerate(depth_fractions):
                x_idx = int(before_data.shape[0] * frac)
                slice_data = before_data[x_idx, :, :].T
                slice_mask = air_mask[x_idx, :, :].T

                axes[1, col].imshow(slice_data, cmap='gray', origin='lower')
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [1, 0, 0, 0.3]
                axes[1, col].imshow(overlay, origin='lower')
                axes[1, col].set_title(f'Sagittal x={x_idx}')
                axes[1, col].axis('off')

            # Coronal slices (row 2)
            for col, frac in enumerate(depth_fractions):
                y_idx = int(before_data.shape[1] * frac)
                slice_data = before_data[:, y_idx, :].T
                slice_mask = air_mask[:, y_idx, :].T

                axes[2, col].imshow(slice_data, cmap='gray', origin='lower')
                overlay = np.zeros((*slice_mask.shape, 4))
                overlay[slice_mask, :] = [1, 0, 0, 0.3]
                axes[2, col].imshow(overlay, origin='lower')
                axes[2, col].set_title(f'Coronal y={y_idx}')
                axes[2, col].axis('off')

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e

