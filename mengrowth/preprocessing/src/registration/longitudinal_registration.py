"""Longitudinal registration implementation using ANTs.

This module implements pairwise registration of images from different
timestamps to a reference timestamp for the same patient.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import time
import shutil
import tempfile

import numpy as np

from mengrowth.preprocessing.src.registration.base import BaseRegistrator

logger = logging.getLogger(__name__)


class LongitudinalRegistration(BaseRegistrator):
    """Longitudinal registration across timestamps using ANTs.

    This class handles pairwise registration of images from different
    timestamps to a reference timestamp for longitudinal analysis.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize longitudinal registration.

        Args:
            config: Configuration dictionary
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)
        self.engine = config.get("engine", "antspyx")

        # Import appropriate backend
        if self.engine == "antspyx":
            try:
                import ants
                self.ants = ants
            except ImportError:
                raise ImportError(
                    "antspyx required but not installed. "
                    "Install with: pip install antspyx"
                )
        else:
            raise ValueError(f"Unsupported engine: {self.engine}. Only 'antspyx' is supported for longitudinal registration.")

    def register_pair(
        self,
        fixed_path: Path,
        moving_path: Path,
        output_path: Path,
        transform_path: Path
    ) -> None:
        """Register a moving image to a fixed image.

        Handles z-scored images (intensity-normalized with mean=0) by shifting
        values to the positive range before registration. ANTs uses center-of-mass
        initialization internally, which fails when total image mass is near zero
        (sum of z-scored voxels ≈ 0). The shift makes COM work; the transform is
        then applied to the original unshifted image so output values are preserved.

        Args:
            fixed_path: Path to fixed (reference) image
            moving_path: Path to moving image to register
            output_path: Path to save registered image
            transform_path: Path to save transform file

        Raises:
            RuntimeError: If registration fails
        """
        logger.debug(f"      Registering {moving_path.name} → {fixed_path.name}")

        try:
            # Load images
            fixed = self.ants.image_read(str(fixed_path))
            moving = self.ants.image_read(str(moving_path))

            # ----------------------------------------------------------
            # Pre-flight: validate images have non-zero content
            # ----------------------------------------------------------
            fixed_data = fixed.numpy()
            moving_data = moving.numpy()

            fixed_nonzero = np.count_nonzero(fixed_data)
            moving_nonzero = np.count_nonzero(moving_data)

            if fixed_nonzero == 0:
                raise RuntimeError(
                    f"Fixed (reference) image is all zeros: {fixed_path}. "
                    "Check upstream steps (skull stripping, atlas registration, "
                    "intensity normalization)."
                )
            if moving_nonzero == 0:
                raise RuntimeError(
                    f"Moving image is all zeros: {moving_path}. "
                    "Check upstream steps (skull stripping, atlas registration, "
                    "intensity normalization)."
                )

            # ----------------------------------------------------------
            # Handle z-scored images: shift to positive range for COM
            # ----------------------------------------------------------
            # After z-score normalization, brain voxels have mean≈0 and
            # background=0. ANTs' ImageMomentsCalculator sums raw voxel
            # values for center-of-mass → total mass ≈ 0 → assertion
            # failure. Shifting to positive range fixes COM while the
            # resulting transform is identical (MI is shift-invariant for
            # the same uniform offset on both images).
            needs_shift = float(fixed_data.min()) < 0 or float(moving_data.min()) < 0

            if needs_shift:
                shift = max(abs(float(fixed_data.min())),
                            abs(float(moving_data.min()))) + 1.0
                logger.debug(
                    f"      Z-scored images detected (min fixed={fixed_data.min():.2f}, "
                    f"min moving={moving_data.min():.2f}). "
                    f"Shifting by +{shift:.1f} for center-of-mass stability."
                )
                fixed_reg = self.ants.from_numpy(
                    fixed_data + shift,
                    origin=fixed.origin,
                    spacing=fixed.spacing,
                    direction=fixed.direction
                )
                moving_reg = self.ants.from_numpy(
                    moving_data + shift,
                    origin=moving.origin,
                    spacing=moving.spacing,
                    direction=moving.direction
                )
            else:
                fixed_reg = fixed
                moving_reg = moving

            # ----------------------------------------------------------
            # Prepare registration parameters
            # ----------------------------------------------------------
            transform_type = self.config.get("transform_type", ["Rigid", "Affine"])
            if isinstance(transform_type, str):
                transform_type = [transform_type]

            # ANTsPy expects a single string. When multiple transforms are
            # specified, use the last (most comprehensive) one — e.g.,
            # ["Rigid", "Affine"] → "Affine" which subsumes Rigid.
            if len(transform_type) == 1:
                type_of_transform = transform_type[0]
            else:
                type_of_transform = transform_type[-1]

            reg_params = {
                "fixed": fixed_reg,
                "moving": moving_reg,
                "type_of_transform": type_of_transform,
                "aff_metric": self.config.get("metric", "Mattes"),
                "aff_sampling": int(self.config.get("sampling_percentage", 0.5) * 32),
                "syn_metric": self.config.get("metric", "Mattes"),
                "syn_sampling": int(self.config.get("sampling_percentage", 0.5) * 32),
                "verbose": self.verbose,
            }

            # Add multi-resolution parameters if available
            if "number_of_iterations" in self.config:
                iterations = self.config["number_of_iterations"]
                if isinstance(transform_type, list) and len(transform_type) > 1:
                    reg_params["aff_iterations"] = iterations[0] if iterations else [1000, 500, 250, 0]
                else:
                    reg_params["aff_iterations"] = iterations[0] if iterations else [1000, 500, 250, 0]

            # ----------------------------------------------------------
            # Run registration
            # ----------------------------------------------------------
            start_time = time.time()
            registration_result = self.ants.registration(**reg_params)
            elapsed_time = time.time() - start_time

            logger.debug(f"      Registration completed in {elapsed_time:.2f}s")

            # ----------------------------------------------------------
            # Save registered image
            # ----------------------------------------------------------
            # When images were shifted, apply the transform to the ORIGINAL
            # (unshifted) moving image so output intensities are preserved.
            if needs_shift:
                warped = self.ants.apply_transforms(
                    fixed=fixed,
                    moving=moving,
                    transformlist=registration_result['fwdtransforms'],
                    interpolator='linear'
                )
            else:
                warped = registration_result['warpedmovout']

            temp_output = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            temp_output.close()
            temp_output_path = Path(temp_output.name)

            try:
                self.ants.image_write(warped, str(temp_output_path))
                shutil.move(str(temp_output_path), str(output_path))
            except Exception as e:
                if temp_output_path.exists():
                    temp_output_path.unlink()
                raise e

            # ----------------------------------------------------------
            # Save transform
            # ----------------------------------------------------------
            if registration_result.get('fwdtransforms'):
                if self.config.get("write_composite_transform", True):
                    if len(registration_result['fwdtransforms']) == 1:
                        shutil.copy(registration_result['fwdtransforms'][0], str(transform_path))
                    else:
                        shutil.copy(registration_result['fwdtransforms'][0], str(transform_path))
                else:
                    for i, tx_file in enumerate(registration_result['fwdtransforms']):
                        tx_out_path = transform_path.parent / f"{transform_path.stem}_{i}{transform_path.suffix}"
                        shutil.copy(tx_file, str(tx_out_path))

            logger.debug(f"      Transform saved: {transform_path.name}")

        except Exception as e:
            raise RuntimeError(f"Longitudinal registration failed: {e}") from e

    def execute(self, *args, **kwargs):
        """Not used for longitudinal registration (uses register_pair instead)."""
        raise NotImplementedError(
            "Longitudinal registration uses register_pair() method. "
            "The execute() method is not applicable for patient-level operations."
        )

    def visualize(
        self,
        reference_path: Path,
        pre_registration_path: Path,
        post_registration_path: Path,
        output_path: Path
    ) -> None:
        """Generate visualization comparing pre and post registration to reference.

        Creates a 3×2 grid visualization:
        - Rows: Axial, Sagittal, Coronal views
        - Column 1: Reference (target) volume in grayscale
        - Column 2: Overlay of pre-registration (jet colormap) with post-registration (gray) on top

        Args:
            reference_path: Path to reference (target) image
            pre_registration_path: Path to pre-registration moving image
            post_registration_path: Path to post-registration (aligned) image
            output_path: Path to save visualization PNG

        Raises:
            RuntimeError: If visualization generation fails
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Headless rendering
            import matplotlib.pyplot as plt
            import nibabel as nib
            import numpy as np

            # Load images
            logger.debug(f"Loading images for visualization")
            reference_img = nib.load(str(reference_path))
            pre_reg_img = nib.load(str(pre_registration_path))
            post_reg_img = nib.load(str(post_registration_path))

            reference_data = reference_img.get_fdata()
            pre_reg_data = pre_reg_img.get_fdata()
            post_reg_data = post_reg_img.get_fdata()

            # Get middle slices for each view
            shape = reference_data.shape
            axial_slice = shape[2] // 2
            sagittal_slice = shape[0] // 2
            coronal_slice = shape[1] // 2

            # Create figure with 3 rows × 2 columns
            fig, axes = plt.subplots(3, 2, figsize=(12, 18))

            # Define views and slices
            views = [
                ("Axial",
                 reference_data[:, :, axial_slice],
                 pre_reg_data[:, :, axial_slice],
                 post_reg_data[:, :, axial_slice]),
                ("Sagittal",
                 reference_data[sagittal_slice, :, :],
                 pre_reg_data[sagittal_slice, :, :],
                 post_reg_data[sagittal_slice, :, :]),
                ("Coronal",
                 reference_data[:, coronal_slice, :],
                 pre_reg_data[:, coronal_slice, :],
                 post_reg_data[:, coronal_slice, :])
            ]

            for row, (view_name, ref_slice, pre_slice, post_slice) in enumerate(views):
                # Column 0: Reference (target) volume
                axes[row, 0].imshow(ref_slice.T, cmap='gray', origin='lower')
                axes[row, 0].set_title(f'{view_name}: Reference (Target)', fontsize=10)
                axes[row, 0].axis('off')

                # Column 1: Overlay of pre-registration (jet) with post-registration (gray) on top
                # First, show pre-registration in jet colormap
                axes[row, 1].imshow(pre_slice.T, cmap='jet', origin='lower', alpha=1.0)
                # Then overlay post-registration in grayscale with some transparency
                axes[row, 1].imshow(post_slice.T, cmap='gray', origin='lower', alpha=0.6)
                axes[row, 1].set_title(
                    f'{view_name}: Pre-reg (jet) + Post-reg (gray overlay)',
                    fontsize=10
                )
                axes[row, 1].axis('off')

            # Add overall title
            fig.suptitle(
                f'Longitudinal Registration Comparison\n'
                f'Reference: {reference_path.name}\n'
                f'Moving: {pre_registration_path.name}',
                fontsize=12,
                y=0.995
            )

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.debug(f"Visualization saved: {output_path}")

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization generation failed: {e}") from e
