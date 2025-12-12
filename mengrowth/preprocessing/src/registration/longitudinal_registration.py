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

            # Prepare registration parameters
            transform_type = self.config.get("transform_type", ["Rigid", "Affine"])
            if isinstance(transform_type, str):
                transform_type = [transform_type]

            # Convert transform types to ANTsPy format
            # ANTsPy expects a string like: "Rigid", "Affine", "SyN"
            # Transform types in ANTs are hierarchical (Affine includes Rigid, SyN includes both)
            # So if multiple transforms are specified, use the last (most comprehensive) one
            if len(transform_type) == 1:
                type_of_transform = transform_type[0]
            else:
                # Use the last transform type as it's the most comprehensive
                # (e.g., ["Rigid", "Affine"] -> "Affine" which includes Rigid)
                type_of_transform = transform_type[-1]

            # Build registration parameters
            reg_params = {
                "fixed": fixed,
                "moving": moving,
                "type_of_transform": type_of_transform,
                "aff_metric": self.config.get("metric", "Mattes"),
                "aff_sampling": int(self.config.get("sampling_percentage", 0.5) * 32),  # ANTs uses sampling value
                "syn_metric": self.config.get("metric", "Mattes"),
                "syn_sampling": int(self.config.get("sampling_percentage", 0.5) * 32),
                "verbose": self.verbose,
            }

            # Add multi-resolution parameters if available
            if "number_of_iterations" in self.config:
                # ANTsPy expects a flat list for rigid/affine, or list of lists for multi-stage
                iterations = self.config["number_of_iterations"]
                if isinstance(transform_type, list) and len(transform_type) > 1:
                    # Multi-stage registration
                    # For now, use the first transform's iterations
                    # ANTsPy will handle multi-stage automatically
                    reg_params["aff_iterations"] = iterations[0] if iterations else [1000, 500, 250, 0]
                else:
                    reg_params["aff_iterations"] = iterations[0] if iterations else [1000, 500, 250, 0]

            # Perform registration
            start_time = time.time()
            registration_result = self.ants.registration(**reg_params)
            elapsed_time = time.time() - start_time

            logger.debug(f"      Registration completed in {elapsed_time:.2f}s")

            # Save registered image (use temp file to avoid corrupting original on failure)
            temp_output = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            temp_output.close()
            temp_output_path = Path(temp_output.name)

            try:
                self.ants.image_write(registration_result['warpedmovout'], str(temp_output_path))

                # If successful, move to final location
                shutil.move(str(temp_output_path), str(output_path))
            except Exception as e:
                # Clean up temp file on error
                if temp_output_path.exists():
                    temp_output_path.unlink()
                raise e

            # Save transform
            # ANTs returns transforms in registration_result['fwdtransforms']
            # This is a list of transform files
            if registration_result.get('fwdtransforms'):
                # For composite transform, we want to save the combined transform
                if self.config.get("write_composite_transform", True):
                    # If there's only one transform file, just copy it
                    if len(registration_result['fwdtransforms']) == 1:
                        shutil.copy(registration_result['fwdtransforms'][0], str(transform_path))
                    else:
                        # ANTsPy composite transforms are typically .h5 or .mat files
                        # Copy the first (composite) transform
                        shutil.copy(registration_result['fwdtransforms'][0], str(transform_path))
                else:
                    # Save individual transforms
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
