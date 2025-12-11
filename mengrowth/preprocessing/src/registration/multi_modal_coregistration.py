"""Multi-modal intra-study coregistration using ANTs.

This module implements rigid registration of multiple MRI modalities
to a reference sequence within the same study/time-point.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import time

import nibabel as nib
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.registration.base import BaseRegistrator

logger = logging.getLogger(__name__)


class MultiModalCoregistration(BaseRegistrator):
    """Intra-study rigid multi-modal coregistration using ANTs.

    For each study, registers all modalities (T1n, T2w, T2-FLAIR, etc.)
    to a reference modality (typically T1c) using rigid registration.

    The reference modality is selected based on a priority list, allowing
    graceful handling of missing modalities.

    Attributes:
        config: Registration configuration parameters
        verbose: Whether to enable verbose logging
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize multi-modal coregistration step.

        Args:
            config: Configuration dictionary from RegistrationConfig
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)
        self.logger = logging.getLogger(__name__)

    def execute(
        self,
        study_dir: Path,
        artifacts_dir: Path,
        modalities: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute multi-modal coregistration for a single study.

        Args:
            study_dir: Directory containing modality files (*.nii.gz)
            artifacts_dir: Directory to save transform artifacts
            modalities: List of expected modalities
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with:
            - reference_modality: Selected reference modality
            - registered_modalities: List of registered modalities
            - transforms: Dict mapping modality to transform path

        Raises:
            ValueError: If fewer than 2 modalities found
            RuntimeError: If registration fails
        """
        start_time = time.time()
        self.logger.info(f"Starting multi-modal coregistration in {study_dir.name}")

        # 1. Discover available modality files
        modality_files = self._discover_modality_files(study_dir, modalities)

        if len(modality_files) < 2:
            raise ValueError(
                f"Registration requires at least 2 modalities, found {len(modality_files)}: "
                f"{list(modality_files.keys())}"
            )

        # 2. Select reference modality
        reference_modality = self._select_reference_modality(
            available_modalities=list(modality_files.keys()),
            priority_str=self.config.get("reference_modality_priority", "t1c > t1n > t2f > t2w")
        )
        reference_path = modality_files[reference_modality]

        self.logger.info(
            f"Reference: {reference_modality}, "
            f"Moving: {[m for m in modality_files if m != reference_modality]}"
        )

        # 3. Create registration artifacts directory
        registration_dir = artifacts_dir / "registration"
        registration_dir.mkdir(parents=True, exist_ok=True)

        # 4. Register each non-reference modality
        transforms = {}
        registered_modalities = []

        for modality, moving_path in modality_files.items():
            if modality == reference_modality:
                continue  # Skip reference modality

            self.logger.info(f"Registering {modality} to {reference_modality}...")

            # Define transform output path
            transform_path = registration_dir / f"{modality}_to_{reference_modality}.h5"

            try:
                # Perform registration
                # _register_modality returns (registered_path, actual_transform_path)
                registered_path, actual_transform = self._register_modality(
                    fixed_path=reference_path,
                    moving_path=moving_path,
                    transform_path=transform_path,
                    study_dir=study_dir,
                    modality=modality
                )

                transforms[modality] = actual_transform
                registered_modalities.append(modality)

                self.logger.info(f"✓ {modality} registered successfully")

            except Exception as e:
                self.logger.error(f"✗ Failed to register {modality}: {e}")
                # Continue with other modalities rather than failing completely
                continue

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed registration in {elapsed:.1f}s. "
            f"Registered {len(registered_modalities)}/{len(modality_files)-1} modalities"
        )

        return {
            "reference_modality": reference_modality,
            "registered_modalities": registered_modalities,
            "transforms": transforms
        }

    def _register_modality(
        self,
        fixed_path: Path,
        moving_path: Path,
        transform_path: Path,
        study_dir: Path,
        modality: str
    ) -> Path:
        """Register a single modality to the reference using ANTs.

        Args:
            fixed_path: Path to reference (fixed) image
            moving_path: Path to moving image
            transform_path: Path to save transform file
            study_dir: Study directory for output
            modality: Modality name

        Returns:
            Tuple of (registered_output_file_path, actual_transform_path)

        Raises:
            RuntimeError: If registration fails
        """
        from nipype.interfaces import ants

        # Create a temporary output path for the registered image
        temp_output = study_dir / f"_temp_{modality}_registered.nii.gz"

        if self.verbose:
            self.logger.debug(f"[DEBUG] Registration setup for {modality}:")
            self.logger.debug(f"  Fixed image:     {fixed_path}")
            self.logger.debug(f"  Moving image:    {moving_path}")
            self.logger.debug(f"  Transform (req): {transform_path}")
            self.logger.debug(f"  Temp output:     {temp_output}")

        # Initialize ANTs Registration interface
        reg = ants.Registration()
        reg.inputs.dimension = 3
        reg.inputs.fixed_image = str(fixed_path)
        reg.inputs.moving_image = str(moving_path)

        # Transform configuration (can be string or list)
        transform_type_config = self.config.get("transform_type", "Rigid")

        # Normalize to list for uniform handling
        if isinstance(transform_type_config, str):
            transforms = [transform_type_config]
        else:
            transforms = transform_type_config

        reg.inputs.transforms = transforms

        # Transform parameters (gradient step for optimization)
        # One parameter tuple per transform
        # For Rigid/Affine: [gradient_step], typical value is 0.1
        reg.inputs.transform_parameters = [(0.1,)] * len(transforms)

        # Metric configuration (one per transform)
        metric = self.config.get("metric", "Mattes")
        metric_bins = self.config.get("metric_bins", 32)
        reg.inputs.metric = [metric] * len(transforms)
        reg.inputs.metric_weight = [1.0] * len(transforms)
        reg.inputs.radius_or_number_of_bins = [metric_bins] * len(transforms)

        # Sampling strategy (one per transform)
        sampling_strategy = self.config.get("sampling_strategy", "Random")
        sampling_percentage = self.config.get("sampling_percentage", 0.2)
        reg.inputs.sampling_strategy = [sampling_strategy] * len(transforms)
        reg.inputs.sampling_percentage = [sampling_percentage] * len(transforms)

        # Multi-resolution schedule
        number_of_iterations = self.config.get("number_of_iterations", [[1000, 500, 250]])
        shrink_factors = self.config.get("shrink_factors", [[4, 2, 1]])
        smoothing_sigmas = self.config.get("smoothing_sigmas", [[2, 1, 0]])

        reg.inputs.number_of_iterations = number_of_iterations
        reg.inputs.shrink_factors = shrink_factors
        reg.inputs.smoothing_sigmas = smoothing_sigmas
        reg.inputs.sigma_units = ["vox"]

        # Convergence (one per transform)
        convergence_threshold = self.config.get("convergence_threshold", 1e-6)
        convergence_window_size = self.config.get("convergence_window_size", 10)
        reg.inputs.convergence_threshold = [convergence_threshold] * len(transforms)
        reg.inputs.convergence_window_size = [convergence_window_size] * len(transforms)

        # Output configuration
        write_composite = self.config.get("write_composite_transform", True)
        reg.inputs.write_composite_transform = write_composite

        # Transform prefix should NOT include extension
        # ANTs will append "Composite.h5" when write_composite_transform=True
        # So if transform_path is "/path/to/t2f_to_t1c.h5", we set prefix to "/path/to/t2f_to_t1c"
        # and ANTs will create "/path/to/t2f_to_t1cComposite.h5"
        transform_prefix = str(transform_path.with_suffix(""))
        reg.inputs.output_transform_prefix = transform_prefix
        reg.inputs.output_warped_image = str(temp_output)

        # Update transform_path to point to the actual output file
        # ANTs appends "Composite.h5" to the prefix
        actual_transform_path = Path(str(transform_prefix) + "Composite.h5")

        # Interpolation
        interpolation = self.config.get("interpolation", "Linear")
        reg.inputs.interpolation = interpolation

        # Verbose output
        reg.inputs.verbose = self.verbose

        if self.verbose:
            self.logger.debug(f"[DEBUG] ANTs parameters:")
            self.logger.debug(f"  Transform: {transform_type}")
            self.logger.debug(f"  Metric: {metric} (bins={metric_bins})")
            self.logger.debug(f"  Sampling: {sampling_strategy} ({sampling_percentage*100}%)")
            self.logger.debug(f"  Iterations: {number_of_iterations}")
            self.logger.debug(f"  Shrink: {shrink_factors}")
            self.logger.debug(f"  Smoothing: {smoothing_sigmas}")
            self.logger.debug(f"  Convergence: {reg.inputs.convergence_threshold}")
            self.logger.debug(f"  Interpolation: {interpolation}")
            self.logger.debug(f"  Transform prefix: {transform_prefix}")
            self.logger.debug(f"  Actual transform file: {actual_transform_path}")

        try:
            # Run registration
            if self.verbose:
                self.logger.debug(f"[DEBUG] Executing ANTs registration...")
                self.logger.debug(f"[DEBUG] Command line will be constructed by Nipype")

                # Try to get the command line before running
                try:
                    test_cmdline = reg.cmdline
                    self.logger.debug(f"[DEBUG] Constructed command line:")
                    self.logger.debug(f"  {test_cmdline}")
                except Exception as cmdline_err:
                    self.logger.debug(f"[DEBUG] Failed to construct command line: {cmdline_err}")
                    import traceback
                    self.logger.debug(f"[DEBUG] Traceback:")
                    for line in traceback.format_exc().split('\n'):
                        self.logger.debug(f"  {line}")

            result = reg.run()

            if self.verbose:
                self.logger.debug(f"[DEBUG] Registration completed successfully")
                self.logger.debug(f"[DEBUG] Checking outputs...")

            # Verify transform was created
            if not actual_transform_path.exists():
                # List files in transform directory to debug
                transform_dir = actual_transform_path.parent
                if self.verbose and transform_dir.exists():
                    self.logger.debug(f"[DEBUG] Files in {transform_dir}:")
                    for f in transform_dir.iterdir():
                        self.logger.debug(f"  - {f.name}")
                raise RuntimeError(f"Transform file not created: {actual_transform_path}")

            # Verify output image was created
            if not temp_output.exists():
                raise RuntimeError(f"Registered image not created: {temp_output}")

            if self.verbose:
                self.logger.debug(f"[DEBUG] Outputs verified successfully")
                self.logger.debug(f"[DEBUG] Replacing {moving_path.name} with registered version")

            # Replace original with registered version (in-place)
            final_output = moving_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(f"[DEBUG] File replacement complete")

            return final_output, actual_transform_path

        except Exception as e:
            # Enhanced error reporting
            error_msg = f"ANTs registration failed for {modality}: {str(e)}"

            if self.verbose:
                self.logger.debug(f"[DEBUG] Registration error details:")
                self.logger.debug(f"  Error type: {type(e).__name__}")
                self.logger.debug(f"  Error message: {str(e)}")

                # Try to get the command line that failed
                try:
                    cmdline = reg.cmdline
                    self.logger.debug(f"  Command line: {cmdline}")
                except:
                    self.logger.debug(f"  Could not retrieve command line")

                # Check if input files exist
                self.logger.debug(f"  Fixed image exists: {fixed_path.exists()}")
                self.logger.debug(f"  Moving image exists: {moving_path.exists()}")

            # Clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()
                if self.verbose:
                    self.logger.debug(f"[DEBUG] Cleaned up temp file: {temp_output}")

            raise RuntimeError(error_msg) from e

    def visualize(
        self,
        reference_path: Path,
        moving_path: Path,
        registered_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing reference, moving, and registered images.

        Creates a 1x3 subplot showing:
        1. Reference (fixed) image
        2. Moving image (before registration)
        3. Registered image (after registration)

        Args:
            reference_path: Path to reference image
            moving_path: Path to moving image (original)
            registered_path: Path to registered image
            output_path: Path to save visualization PNG
            **kwargs: Additional parameters (transform_path, modality)
        """
        try:
            # Load images
            ref_img = nib.load(str(reference_path))
            mov_img = nib.load(str(moving_path))
            reg_img = nib.load(str(registered_path))

            ref_data = ref_img.get_fdata()
            mov_data = mov_img.get_fdata()
            reg_data = reg_img.get_fdata()

            # Select middle slice (axial view)
            slice_idx = ref_data.shape[2] // 2

            ref_slice = ref_data[:, :, slice_idx]
            mov_slice = mov_data[:, :, slice_idx]
            reg_slice = reg_data[:, :, slice_idx]

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Reference image
            axes[0].imshow(ref_slice.T, cmap="gray", origin="lower")
            axes[0].set_title("Reference (Fixed)")
            axes[0].axis("off")

            # Moving image
            axes[1].imshow(mov_slice.T, cmap="gray", origin="lower")
            axes[1].set_title("Moving (Original)")
            axes[1].axis("off")

            # Registered image
            axes[2].imshow(reg_slice.T, cmap="gray", origin="lower")
            axes[2].set_title("Registered")
            axes[2].axis("off")

            # Add metadata
            modality = kwargs.get("modality", "unknown")
            transform_path = kwargs.get("transform_path", "N/A")

            fig.suptitle(
                f"Multi-Modal Coregistration: {modality}\n"
                f"Transform: {transform_path.name if hasattr(transform_path, 'name') else transform_path}",
                fontsize=12
            )

            plt.tight_layout()

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved visualization: {output_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
