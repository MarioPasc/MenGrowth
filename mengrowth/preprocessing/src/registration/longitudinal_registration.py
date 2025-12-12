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
            # ANTsPy expects format like: "Rigid", "Affine", "SyN"
            # For multiple transforms, it concatenates them
            if len(transform_type) == 1:
                type_of_transform = transform_type[0]
            else:
                # For multiple transforms, ANTsPy can handle a list
                # But we'll use the composite approach
                type_of_transform = transform_type

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

    def visualize(self, *args, **kwargs):
        """Visualization for longitudinal registration.

        TODO: Implement visualization similar to other registration steps.
        Could show before/after registration, checkerboard overlay, etc.
        """
        # Optional: implement visualization similar to other registration steps
        logger.debug("Visualization not yet implemented for longitudinal registration")
        pass
